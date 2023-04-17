from PIL import Image
from torchvision.transforms import transforms
import torchvision.transforms as Transforms

from model.our import  UNet2
# from utils.Evaluation import Evaluation
# from utils.dataset import ISBI_Loader
from torch import optim
# from tensorboardX import SummaryWriter
from torchvision import utils,transforms
import torch.utils.data as data
import time
import glob
import cv2 as cv
import cv2

from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn

from model.our1 import UNet3
from util.dataset3 import  ISBI_Loader

def wbce_loss(y_pred, label ,alpha):
   # 计算loss
   p = torch.sigmoid(y_pred)

   # p = torch.clamp(p, min=1e-9, max=0.99)

   # loss = torch.sum(torch.abs_(-alpha*label*torch.log(p)+(1-label)*torch.log(1-p))) / len(label)
   loss = torch.sum(- alpha * torch.log(p) * label \
                    - torch.log(1 - p) * (1 - label)) / ((32 ** 2)*3)
   return loss

def train_net(net, device, data_path, epochs=20, batch_size=3, lr=0.0001, ModelName='FC_EF', is_Transfer= False):
    print('Conrently, Traning Model is :::::'+ModelName+':::::')
    if is_Transfer:
        print("Loading Transfer Learning M   odel.........")
        # BFENet.load_state_dict(torch.load('Pretrain_BFE_'+ModelName+'_model_epoch75_mIoU_89.657089.pth', map_location=device))
    else:
        print("No Using Transfer Le arning Model.........")

    # 加载数据集
    isbi_dataset = ISBI_Loader(data_path=data_path, transform=Transforms.ToTensor())
    train_loader = data.DataLoader(dataset=isbi_dataset,
                                   batch_size=batch_size,
                                   shuffle=False)
    # 定义RMSprop算法
    # optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    # scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70], gamma=0.9)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90], gamma=0.9)

    # 定义loss
    # weight = torch.tensor([5], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss()
    #criterion = CDWithLogitsLoss(device, beta_EL=0, beta_FL=0, beta_BCE=1)
    # criterion = BCEFocalLoss(device)
    # best loss, 初始化为正无穷
    # best_loss = float('inf')
    # writer = SummaryWriter('runs/exp')
    f_loss = open('txt\\train_loss.txt', 'w')
    f_time = open('txt\\train_time.txt', 'w')
    # 训练epochs次
    i = 0

    start = time.time()
    for epoch in range(1, epochs+1):
        net.train()
        # 训练模式
        # learning rate delay
        best_loss = float('inf')
        best_F1 = float('inf')
        # 按照batch_size开始训练
        n = 0
        num=0
        total=0
        starttime = time.time()
        print('==========================epoch = '+str(epoch)+'==========================')
        for image1, image2,label in train_loader:
            optimizer.zero_grad()
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            m=0
            # 将数据拷贝到device中
            image1 = image1.to(device=device)
            image2 = image2.to(device=device)


            label = label.to(device=device)
            # print(label.shape)
            # 使用网络参数，输出预测结果
            # list = []   # 0: out1,1: out2,2: feat1,3: feat2

            pred = net(image1,image2)
            # pred_img=np.array(pred.data.cpu()[0])[0]
            # i=i+1
            # pred_img[pred_img >= 0.5] = 255
            # pred_img[pred_img < 0.5] = 0
            #
            #
            # img = label.mul(255).byte()
            # label1 = img.numpy().squeeze(0).transpose((1, 2, 0))
            #
            #
            #
            # cv2.imwrite("data/LEVIR-CD/pred/res_"+str(i)+".png", label1)
            # label1=cv2.imread("data/LEVIR-CD/pred/res_"+str(i)+".png")
            # label1 = cv2.cvtColor(label1, cv2.COLOR_BGR2GRAY)
            # Indicators = Evaluation(label1, pred_img)
            # Indicators = Indicators.ConfusionMatrix()
            # print(Indicators[0], "  ", Indicators[1], "  ", Indicators[2], "  ", Indicators[3])
            # m= (Indicators[1] +Indicators[3])/(1024**2)
            # print(m)
            #1
            total_loss = criterion(pred, label)


            #2

            # pred.requires_grad_(True)
            # pred=pred.detach()
            # loss_ = wbce_loss(pred, label,1)
            #3
            # loss_ = focal_loss(label, pred)
            # loss_1 = dice_loss(pred, label)
            # loss_ = 0.5 * loss_1 + (1 -0.5) * loss_
            total+=total_loss.item()

            # if num == 0:
            #     if epoch == 0:
            #         f_loss.write('Note: epoch (num, edge_loss, focal_loss, BCE_loss, total_loss)\n')
            #         f_loss.write('epoch = ' + str(epoch) + '\n')
            #     else:
            #         f_loss.write('epoch = ' + str(epoch) + '\n')
            # f_loss.write(str(num) + ',' + str(float('%5f' % total_loss)) + '\n')
            # writer.add_scalar('epoch', total_loss, global_step=epoch)
            # writer.add_scalar('train_total_num', total_loss, global_step=num)
            print(str(epoch)+'/' + str(epochs)+':::::'+'lr='+str(optimizer.param_groups[0]['lr'])+':::::'+str(num)+'/'+str(int(len(isbi_dataset)/batch_size)))
            print('Loss/train', total_loss.item())
            # print(loss_.item())
            # print(loss1.item())
            print('-----------------------------------------------------------------------')
            # 保存loss值最小的网络参数
            """
            if epoch % 10 == 0:
                if total_loss < best_loss:
                    best_loss = total_loss
                    BFE_path = 'best_BFE_SPM_model_epoch' + str(epoch) + '.pth'
                    BCD_path = 'best_BCD_SPM_model_epoch' + str(epoch) + '.pth'
                    torch.save(BFENet.state_dict(), BFE_path)
                    torch.save(net.state_dict(), BCD_path)
            """
            # 更新参数
            #1
            total_loss.backward()
            #2,3
            # loss_.backward()

            optimizer.step()

            n += 1
        # learning rate delay

        f_loss.write(str(total/n)+'\n')
        scheduler1.step()
        torch.save(net.state_dict(), 'best_BFE_1_model_finalshu.pth')
    endtime = time.time()
    f_loss.write(str(endtime-start) + '\n')




if __name__ == '__main__':
    # 选择设备，有cuda用cuda，没有就用cpu
    device = 'cpu'  # tor ch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道3，分类1(目标)
    # net = SiamUnet_conc(input_nbr=3, label_nbr=1)
    # net = SiamUnet_diff(input_nbr=3, label_nbr=1)

    # net=MUNet(n_channels=6, n_classes=1)
    # 将网络拷贝到device中
    net = UNet2(n_channels=6, n_classes=1)
    # net = Our(n_channels=6, n_classes=1)


    # net = FMUnet(n_channels=3, n_classes=1)
    net.to(device=device)
    # 指定训练集地址，开始训练
    # data_path = "./samples/LEVIR/train"
    # data_path = "./samples/WHU/train"
    data_path = "./data/Landsat/"
    train_net(net, device, data_path)