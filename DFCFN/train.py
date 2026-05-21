import os
import argparse
import json
import torch
import numpy as np
from torch.nn.functional import threshold, unfold
from dataloaders.HSI_datasets import *
from utils.logger import Logger
import torch.utils.data as data
from utils.helpers import initialize_weights, initialize_weights_new, to_variable, make_patches
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from models.models import MODELS
from utils.metrics import *
import shutil
import torchvision
from torch.distributions.uniform import Uniform
import sys
#import kornia
#from kornia import laplacian, sobel
from scipy.io import savemat
import torch.nn.functional as F
from utils.vgg_perceptual_loss import VGGPerceptualLoss, VGG19
from utils.spatial_loss import Spatial_Loss

''' \\ -2023-0323 add - \\'''
from visdom import Visdom
import numpy as np
import time
from tqdm import tqdm

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

__dataset__ = {"pavia_dataset": pavia_dataset, "paviaU_dataset":paviaU_dataset, "chikusei_dataset": chikusei_dataset, "botswana4_dataset": botswana4_dataset}

# 超参数文件路径
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-c', '--config', default='configs/config_PANNET.json', type=str, help='Path to the config file')
parser.add_argument('-r', '--resume', default=None, type=str, help='Path to the .pth model checkpoint to resume training')
parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
parser.add_argument('--local', action='store_true', default=False)
args = parser.parse_args()

# 加载配置文件
config = json.load(open(args.config))
torch.backends.cudnn.benchmark = True

# 设置种子
torch.manual_seed(7)

# 设置可用于训练的gpu数量
num_gpus = torch.cuda.device_count()

# 模型选择(预训练或正式训练)
model = MODELS[config["model"]](config)
print(f'\n{model}\n')

# 单卡或者多卡训练
if num_gpus > 1:
    print("Training with multiple GPUs ({})".format(num_gpus))
    model = nn.DataParallel(model).cuda()
else:
    print("Single Cuda Node is avaiable")
    model.cuda()

# 建立训练和测试数据存档
print("Training with dataset => {}".format(config["train_dataset"]))
train_loader = data.DataLoader(__dataset__[config["train_dataset"]](config, is_train=True, want_DHP_MS_HR=config["is_DHP_MS"],),
                                batch_size=config["train_batch_size"],
                                num_workers=config["num_workers"],
                                shuffle=True,
                                pin_memory=False,)

test_loader = data.DataLoader(__dataset__[config["train_dataset"]](config,is_train=False,want_DHP_MS_HR=config["is_DHP_MS"],),
                                batch_size=config["val_batch_size"],
                                num_workers=config["num_workers"],
                                shuffle=True,
                                pin_memory=False,)

# 初始化超参数
start_epoch = 1
total_epochs = config["trainer"]["total_epochs"]

# 设置优化器
if config["optimizer"]["type"] == "SGD":
    optimizer = optim.SGD(  model.parameters(), 
                            lr=config["optimizer"]["args"]["lr"], 
                            momentum = config["optimizer"]["args"]["momentum"], 
                            weight_decay= config["optimizer"]["args"]["weight_decay"])
elif config["optimizer"]["type"] == "ADAM":
    optimizer = optim.Adam( model.parameters(), 
                            lr=config["optimizer"]["args"]["lr"],
                            weight_decay= config["optimizer"]["args"]["weight_decay"])
else:
    exit("Undefined optimizer type")

# 学习率分配器。(step_size: 每训练step_size个epoch，更新一次参数；)
scheduler = optim.lr_scheduler.StepLR(  optimizer, 
                                        step_size=config["optimizer"]["step_size"], 
                                        gamma=config["optimizer"]["gamma"])

# 加载训练好的模型或者预训练模型
if args.resume is not None:
    print("Loading from existing FCN and copying weights to continue....")
    checkpoint = torch.load(args.resume)
    '''
    checkpoint.pop("low_cross_attention.temperature")
    checkpoint.pop("high_hor_cross_attention.temperature")
    checkpoint.pop("high_ver_cross_attention.temperature")
    checkpoint.pop("high_dia_cross_attention.temperature")
    checkpoint.pop("cross_attention_02.temperature")
    checkpoint.pop("cross_attention_01.temperature")
    '''
    
    model.load_state_dict(checkpoint, strict=False)
else:
    # initialize_weights(model)
    initialize_weights_new(model)

# 建立损失函数
if config[config["train_dataset"]]["loss_type"] == "L1":
    criterion   = torch.nn.L1Loss()
    HF_loss     = torch.nn.L1Loss()
elif config[config["train_dataset"]]["loss_type"] == "MSE":
    criterion   = torch.nn.MSELoss()
    HF_loss     = torch.nn.MSELoss()
else:
    exit("Undefined loss type")

if config[config["train_dataset"]]["VGG_Loss"]:
    vggnet = VGG19()
    vggnet = torch.nn.DataParallel(vggnet).cuda()

#if config[config["train_dataset"]]["Spatial_Loss"]:
    #Spatial_loss = Spatial_Loss(in_channels = config[config["train_dataset"]]["spectral_bands"]).cuda()

# 训练函数
def train(epoch):
    train_loss = 0.0
    model.train()
    optimizer.zero_grad()
    for i, data in enumerate(train_loader, 0):
    #for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), leave=True):
        # 读取数据
        #_, MS_image, PAN_image, reference = data
        _, MS_image, PAN_image, reference, MS_image_up = data   ### \\ -2023-0315 rewrite - \\

        # 为训练制作较小的patches, 但是json文件中预训练和正式训练is_small_patch_train都设置为false
        if config["trainer"]["is_small_patch_train"]:
            MS_image,_ = make_patches(MS_image, patch_size=config["trainer"]["patch_size"])
            MS_image_up,_ = make_patches(MS_image_up, patch_size=config["trainer"]["patch_size"])   ### \\ -2023-0315 add - \\
            PAN_image,_ = make_patches(PAN_image, patch_size=config["trainer"]["patch_size"])
            reference,_ = make_patches(reference, patch_size=config["trainer"]["patch_size"])

        # 取model输出…
        MS_image    = Variable(MS_image.float().cuda()) 
        MS_image_up    = Variable(MS_image_up.float().cuda())   ### \\ -2023-0315 add - \\ 
        PAN_image   = Variable(PAN_image.float().cuda()) 
        #out         = model(MS_image, PAN_image)
        out         = model(MS_image, PAN_image, MS_image_up)   ### \\ -2023-0315 rewrite - \\
        outputs = out["pred"]

        '''计算损失'''
        # 重建损失
        if config[config["train_dataset"]]["Normalized_L1"]:
            max_ref = torch.amax(reference, dim=(2,3)).unsqueeze(2).unsqueeze(3).expand_as(reference).cuda()
            loss = criterion(outputs/max_ref, to_variable(reference)/max_ref)
        else:
            loss = criterion(outputs, to_variable(reference))

        # VGG感知损失
        if config[config["train_dataset"]]["VGG_Loss"]:
            predicted_RGB = torch.cat((torch.mean(outputs[:, 0:config[config["train_dataset"]]["G"], :, :], 1).unsqueeze(1), 
                                        torch.mean(outputs[:, config[config["train_dataset"]]["B"]:config[config["train_dataset"]]["R"], :, :], 1).unsqueeze(1), 
                                        torch.mean(outputs[:, config[config["train_dataset"]]["G"]:config[config["train_dataset"]]["spectral_bands"], :, :], 1).unsqueeze(1)), 1)
            target_RGB = torch.cat((torch.mean(to_variable(reference)[:, 0:config[config["train_dataset"]]["G"], :, :], 1).unsqueeze(1), 
                                    torch.mean(to_variable(reference)[:, config[config["train_dataset"]]["B"]:config[config["train_dataset"]]["R"], :, :], 1).unsqueeze(1), 
                                    torch.mean(to_variable(reference)[:, config[config["train_dataset"]]["G"]:config[config["train_dataset"]]["spectral_bands"], :, :], 1).unsqueeze(1)), 1)
            VGG_loss = VGGPerceptualLoss(predicted_RGB, target_RGB, vggnet)
            loss += config[config["train_dataset"]]["VGG_Loss_F"]*VGG_loss

        # 转移感知损失
        if config[config["train_dataset"]]["Transfer_Periferal_Loss"]:
            loss += config[config["train_dataset"]]["Transfer_Periferal_Loss_F"]*out["TP_LOSS"]

        # Spatial loss
        #if config[config["train_dataset"]]["Spatial_Loss"]:
            #loss += config[config["train_dataset"]]["Spatial_Loss_F"]*Spatial_loss(to_variable(reference), outputs)
        # Spatial loss
        #if config[config["train_dataset"]]["multi_scale_loss"]:
            #loss += config[config["train_dataset"]]["multi_scale_loss_F"]*criterion(to_variable(reference), out["x13"]) + 2*config[config["train_dataset"]]["multi_scale_loss_F"]*criterion(to_variable(reference), out["x23"])

        torch.autograd.backward(loss)
        
        #网络更新
        if i % config["trainer"]["iter_size"] == 0 or i == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

    writer.add_scalar('Loss/train', loss, epoch)
    
# Testing epoch.
def test(epoch):
    test_loss   = 0.0
    cc          = 0.0
    sam         = 0.0
    rmse        = 0.0
    ergas       = 0.0
    psnr        = 0.0
    val_outputs = {}
    model.eval()
    pred_dic = {}
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader), leave=True):
            #image_dict, MS_image, PAN_image, reference = data
            image_dict, MS_image, PAN_image, reference, MS_image_up = data   ### \\ -2023-0315 rewrite - \\

            # 生成小patches
            if config["trainer"]["is_small_patch_train"]:
                MS_image, unfold_shape = make_patches(MS_image, patch_size=config["trainer"]["patch_size"])
                MS_image_up, unfold_shape = make_patches(MS_image_up, patch_size=config["trainer"]["patch_size"])   ### \\ -2023-0315 add - \\
                PAN_image, _ = make_patches(PAN_image, patch_size=config["trainer"]["patch_size"])
                reference, _ = make_patches(reference, patch_size=config["trainer"]["patch_size"])

            #MS_image    = MS_image.float().cuda()
            #MS_image_up    = MS_image_up.float().cuda()   ### \\ -2023-0315 add - \\
            #PAN_image   = PAN_image.float().cuda()
            #reference   = reference.float().cuda()
            
            MS_image    = Variable(MS_image.float().cuda()) 
            MS_image_up    = Variable(MS_image_up.float().cuda())   ### \\ -2023-0315 add - \\ 
            PAN_image   = Variable(PAN_image.float().cuda()) 
            reference   = Variable(reference.float().cuda())
            # 获取model输出
            #out     = model(MS_image, PAN_image)
            out     = model(MS_image, PAN_image, MS_image_up)   ### \\ -2023-0315 rewrite - \\
            outputs = out["pred"]

            # 计算验证损失
            loss        = criterion(outputs, reference)
            test_loss   += loss.item()

            ''' Scalling 这块是干啥的？'''
            outputs[outputs<0]      = 0.0
            outputs[outputs>1.0]    = 1.0
            outputs                 = torch.round(outputs*config[config["train_dataset"]]["max_value"])
            pred_dic.update({image_dict["imgs"][0].split("/")[-1][:-4]+"_pred": torch.squeeze(outputs).permute(1,2,0).cpu().numpy()})
            reference               = torch.round(reference.detach()*config[config["train_dataset"]]["max_value"])

            '''计算性能指标'''
            cc += cross_correlation(outputs, reference) # Cross-correlation
            sam += SAM(outputs, reference)  # SAM
            rmse += RMSE(outputs/torch.max(reference), reference/torch.max(reference))  # RMSE
            beta = torch.tensor(config[config["train_dataset"]]["HR_size"]/config[config["train_dataset"]]["LR_size"]).cuda()   # ERGAS
            ergas += ERGAS(outputs, reference, beta)
            psnr += PSNR(outputs, reference)    # PSNR

    # 对测试集的性能指标取平均值
    cc /= len(test_loader)
    sam /= len(test_loader)
    rmse /= len(test_loader)
    ergas /= len(test_loader)
    psnr /= len(test_loader)

    # 将测试结果写入tensorboard
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Test_Metrics/CC', cc, epoch)
    writer.add_scalar('Test_Metrics/SAM', sam, epoch)
    writer.add_scalar('Test_Metrics/RMSE', rmse, epoch)
    writer.add_scalar('Test_Metrics/ERGAS', ergas, epoch)
    writer.add_scalar('Test_Metrics/PSNR', psnr, epoch)

    # 图像进tensorboard
    # 重新生成最终图像 
    '''要不要重建一下 MS_image_up '''
    if config["trainer"]["is_small_patch_train"]:
        outputs = outputs.view(unfold_shape).permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        outputs = outputs.contiguous().view(config["val_batch_size"], 
                                            config[config["train_dataset"]]["spectral_bands"],
                                            config[config["train_dataset"]]["HR_size"],
                                            config[config["train_dataset"]]["HR_size"])
        reference = reference.view(unfold_shape).permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        reference = reference.contiguous().view(config["val_batch_size"], 
                                                config[config["train_dataset"]]["spectral_bands"],
                                                config[config["train_dataset"]]["HR_size"],
                                                config[config["train_dataset"]]["HR_size"])
        MS_image = MS_image.view(unfold_shape).permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        MS_image = MS_image.contiguous().view(config["val_batch_size"], 
                                                config[config["train_dataset"]]["spectral_bands"],
                                                config[config["train_dataset"]]["HR_size"],
                                                config[config["train_dataset"]]["HR_size"])
    
    #标准化图像
    outputs = outputs/torch.max(reference)
    reference = reference/torch.max(reference)
    MS_image = MS_image/torch.max(reference)
    
    #这里要插值MS_image到原尺寸图像！！！！！！！！！！！！！！！
    if config["model"]=="HyperPNN" or config["is_DHP_MS"]==False:
        MS_image =  F.interpolate(MS_image, scale_factor=(config[config["train_dataset"]]["factor"],config[config["train_dataset"]]["factor"]),mode ='bilinear')
    
    ms      = torch.unsqueeze(MS_image.view(-1, MS_image.shape[-2], MS_image.shape[-1]), 1)
    pred    = torch.unsqueeze(outputs.view(-1, outputs.shape[-2], outputs.shape[-1]), 1)
    ref     = torch.unsqueeze(reference.view(-1, reference.shape[-2], reference.shape[-1]), 1)
    imgs    = torch.zeros(5*pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3])
    for i in range(pred.shape[0]):
        imgs[5*i]   = ms[i]
        imgs[5*i+1] = torch.abs(ms[i]-pred[i])/torch.max(torch.abs(ms[i]-pred[i]))
        imgs[5*i+2] = pred[i]
        imgs[5*i+3] = ref[i]
        imgs[5*i+4] = torch.abs(ref[i]-ms[i])/torch.max(torch.abs(ref[i]-ms[i]))
    imgs = torchvision.utils.make_grid(imgs, nrow=5)
    writer.add_image('Images', imgs, epoch)

    #返回指标输出
    metrics = { "loss": float(test_loss), 
                "cc": float(cc), 
                "sam": float(sam), 
                "rmse": float(rmse), 
                "ergas": float(ergas), 
                "psnr": float(psnr)}
    print("LOSS: %.6f\tCC: %.6f\tSAM: %.6f\tRMSE: %.6f\tERGAS: %.6f\tPSNR:%.6f" %(test_loss, cc, sam, rmse, ergas, psnr))                
    return image_dict, pred_dic, metrics


'''以下是主程序入口'''
#if __name__ == '__main__':
# 设置tensorboard并复制.json文件保存目录。
PATH = "./"+config["experim_name"]+"/"+config["train_dataset"]+"/"+"N_modules("+str(config["N_modules"])+")"
ensure_dir(PATH+"/")
writer = SummaryWriter(log_dir=PATH)
shutil.copy2(args.config, PATH)

# 打印模型到文本文件
original_stdout = sys.stdout 
with open(PATH+"/"+"model_summary.txt", 'w+') as f:
    sys.stdout = f
    print(f'\n{model}\n')
    sys.stdout = original_stdout 

# 主循环
viz = Visdom()	### \\ -2023-0323 add - \\（实例化监听窗口类）
viz.line([0.], [0], win='TRAIN_SAM', opts=dict(title='TRAIN_SAM', legend=['SAM']))	### \\ -2023-0323 add - \\（创建监听窗口）
viz.line([0.], [0], win='TRAIN_PSNR', opts=dict(title='TRAIN_PSNR', legend=['PSNR']))	### \\ -2023-0323 add - \\（创建监听窗口）
#viz.line([0.], [0], win='TRAIN', opts=dict(title='TRAIN_LOSS', legend=['LOSS']))	### \\ -2023-0323 add - \\（创建监听窗口）
#best_sam = 100.0
best_psnr = 0.0
#best_loss = 1000000
for epoch in range(start_epoch, total_epochs):
    scheduler.step(epoch)
    print("\nTraining Epoch: %d" % epoch)
    #训练
    train(epoch)

    if epoch % config["trainer"]["test_freq"] == 0:
        print("\nTesting Epoch: %d" % epoch)
        #测试
        image_dict, pred_dic, metrics=test(epoch)
        plt_PSNR = metrics["psnr"]	### \\ -2023-0323 add - \\
        viz.line([plt_PSNR], [epoch], win='TRAIN_PSNR', update='append')	### \\ -2023-0323 add - \\
        time.sleep(0.5)	### \\ -2023-0323 add - \\
        
        plt_SAM = metrics["sam"]	### \\ -2023-0323 add - \\
        viz.line([plt_SAM], [epoch], win='TRAIN_SAM', update='append')	### \\ -2023-0323 add - \\
        time.sleep(0.5)	### \\ -2023-0323 add - \\

        #plt_LOSS = metrics["loss"]	### \\ -2023-0323 add - \\
        #viz.line([plt_LOSS], [epoch], win='TRAIN', update='append')	### \\ -2023-0323 add - \\
        #time.sleep(0.5)	### \\ -2023-0323 add - \\
                
        #保存最好的模型
        #if metrics["sam"] < best_sam:
            #best_sam = metrics["sam"]
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
        #if metrics["loss"] < best_loss:
            #best_loss = metrics["loss"]
            #保存最佳性能指标s
            torch.save(model.state_dict(), PATH+"/"+"best_model.pth")
            with open(PATH+"/"+"best_metrics.json", "w+") as outfile: 
                json.dump(metrics, outfile)
            #保存最好的预测
            savemat(PATH+"/"+ "final_prediction.mat", pred_dic)
    
