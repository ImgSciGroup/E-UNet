from model.our1 import UNet3
from util.Evaluation import Evaluation
from util.EvaluationNew import Index
import torchvision.transforms as Transforms
import time
import glob
import cv2
import numpy as np
import torch
from model.our import  UNet2
if __name__ == "__main__":
    print('Starting test...')
    # 选择设备，有cuda用cuda，没有就用cpu
    device = 'cpu'

    # 加载网络，图片3通道，分类为1。
    # Net = SiamUnet_conc(input_nbr=3, label_nbr=1)
    # Net = SiamUnet_diff(input_nbr=3, label_nbr=1)
    # Net = FMUnet(n_channels=3, n_classes=1)
    # Net = UNet3(n_channels=3, n_classes=1)
    Net = UNet2(n_channels=6, n_classes=1)
    # Net = MUNet(n_channels=6, n_classes=1)
    # Net = New(1, 1)
    # 将网络拷贝到device
    Net.to(device=device)

    # 加载模型参数best_BFE_FC_EF_model_final1.pth最好
    Net.load_state_dict(torch.load('best_BFE_1_model_finalshu.pth', map_location=device))

    # 测试模式
    Net.eval()

    trans = Transforms.Compose([Transforms.ToTensor()])
    # 读取所有图片路径
    tests1_path = glob.glob('./data/Landsat/pred/image1/*.tif')
    tests2_path = glob.glob('./data/Landsat/pred/image2/*.tif')
    # tests3_path = glob.glob('./data/Landsat/pred/image3/*.tif')
    label_path = glob.glob('./data/Landsat/pred/label/*.bmp')
    # 遍历所有图片
    num = 1
    TPSum = 0
    TNSum = 0
    FPSum = 0
    FNSum = 0
    C_Sum_or = 0
    UC_Sum_or = 0
    f_acc = open('txt/test_acc.txt', 'w')
    f_time = open('txt/test_time.txt', 'w')
    for tests1_path, tests2_path,label_path in zip(tests1_path,tests2_path, label_path):
        starttime = time.time()
        # 保存结果地址
        save_res_path = '.' + tests1_path.split('.')[1] + '.jpg'

        save_res_path = save_res_path.replace('image1', 'results')
        name = tests1_path.split('/')[4].split('.')[0]

        # 读取图片
        test1_img = cv2.imread(tests1_path)
        test2_img = cv2.imread(tests2_path)
        # test3_img = cv2.imread(tests3_path)

        label_img = cv2.imread(label_path)
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
        test1_img = trans(test1_img)
        test2_img = trans(test2_img)
        # test3_img = trans(test3_img)

        test1_img = test1_img.unsqueeze(0)
        test2_img = test2_img.unsqueeze(0)
        # test3_img = test3_img.unsqueeze(0)

        test1_img = test1_img.to(device=device, dtype=torch.float32)
        test2_img = test2_img.to(device=device, dtype=torch.float32)
        # test3_img = test3_img.to(device=device, dtype=torch.float32)

        # 将tensor拷贝到device中：有gpu就拷贝到gpu，否则就拷贝到cpu
        # 预测
        # 使用网络参数，输出预测结果
        pred_Img = Net(test1_img,test2_img)
        pred1=torch.sigmoid(pred_Img)
        # 提取结果
        pred = np.array(pred_Img.data.cpu()[0])[0]
        pred1= np.array(pred1.data.cpu()[0])[0]
        print(pred1.shape)
        # 处理结果
        pred1 = np.uint8(pred1 * 255.0)
        cv2.imwrite("data/Landsat/pred/CMI/1.png",pred1)
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        print(num, tests1_path)
        # 保存图片
        cv2.imwrite(save_res_path, pred)
        endtime = time.time()
        if num == 0:
            f_time.write('each pair images time\n')
        f_time.write(str(num) + ',' + str(starttime) + ',' + str(endtime) + ',' + str(
            float('%2f' % (starttime - endtime))) + '\n')
        # 评价精度
        monfusion_matrix = Evaluation(label=label_img, pred=pred)
        TP, TN, FP, FN, c_num_or, uc_num_or = monfusion_matrix.ConfusionMatrix()
        TPSum += TP
        TNSum += TN
        FPSum += FP
        FNSum += FN
        C_Sum_or += c_num_or
        UC_Sum_or += uc_num_or
        # 保存验证集loss和accuracy
        if num == 1:
            f_acc.write('=================================================================================\n')
            f_acc.write('|Note: (num, FileName, TP, TN, FP, FN)|\n')
            f_acc.write('|Note: (ACC: FileName, OA, FA, MA, TE, mIoU, c_IoU, uc_IoU, Precision, Recall, F1)|\n')
            f_acc.write('=================================================================================\n')

        f_acc.write(str(num) + ',' + str(name) + '.tif' + ',' + str(TP) + ',' + str(TN) + ',' +
                    str(FP) + ',' + str(FN) + '\n')

        num += 1
        if num % 10 == 0:
            Indicators = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
            IoU, c_IoU, uc_IoU = Indicators.IOU_indicator()
            OA, Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
            FA, MA, TE = Indicators.CD_indicators()

            print("OA=", str(float('%4f' % OA)), "^^^^^", "mIoU=", str(float('%4f' % IoU)), "^^^^^", "c_mIoU=",
                  str(float('%4f' % c_IoU)), "^^^^^", "uc_mIoU=", str(float('%4f' % uc_IoU)), "^^^^^", "Precision=",
                  str(float('%4f' % Precision)), "^^^^^", "Recall=", str(float('%4f' % Recall)), "^^^^^", "mF1=",
                  str(float('%4f' % F1)))

    Indicators = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
    IoU, c_IoU, uc_IoU = Indicators.IOU_indicator()
    OA, Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
    FA, MA, TE = Indicators.CD_indicators()
    print("OA=", str(float('%4f' % OA)), "^^^^^", "mIoU=", str(float('%4f' % IoU)), "^^^^^", "c_mIoU=",
          str(float('%4f' % c_IoU)), "^^^^^", "uc_mIoU=", str(float('%4f' % uc_IoU)), "^^^^^", "Precision=",
          str(float('%4f' % Precision)), "^^^^^", "Recall=", str(float('%4f' % Recall)), "^^^^^", "mF1=",
          str(float('%4f' % F1)))
    f_acc.write(
        '==========================================================================================================\n')
    f_acc.write('|SumConfusionMatrix:|  TP   |   TN   |  FP  |  FN   |\n')
    f_acc.write('|SumConfusionMatrix:|' + str(TPSum) + '|' + str(TNSum) + '|' + str(FPSum) + '|' + str(FNSum) + '|\n')
    f_acc.write(
        '==========================================================================================================\n')
    f_acc.write(
        '|TotalAcc:|   OA   |   FA   |   MA    |  TE   |  mIoU   |  c_IoU  | uc_IoU  |Precision| Recall  |   F1    |\n')
    f_acc.write('|TotalAcc:|' + str(float('%4f' % OA)) + '|' + str(float('%4f' % FA)) + '|' + str(
        float('%4f' % MA)) + '|' + str(float('%4f' % TE))
                + '|' + str(float('%4f' % IoU)) + '|' + str(float('%4f' % c_IoU)) + '|' + str(
        float('%4f' % uc_IoU)) + '|' +
                str(float('%4f' % Precision)) + '|' + str(float('%4f' % Recall)) + '|' + str(float('%4f' % F1)) + '|\n')
    f_acc.write(
        '==========================================================================================================\n')
    f_acc.close()
    f_time.close()