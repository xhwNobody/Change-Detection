import os
import cv2
import torch
import numpy as np
from build_model import Simple_CNN

def bilary2pic(data):
    datasize = data.shape
    for i in range(datasize[0]):
        for j in range(datasize[1]):
            if data[i][j] == 1:
                data[i][j] = 255
    data = np.expand_dims(data, -1)
    result = np.concatenate([data, data, data], -1)
    return result

test_img_path = '/home/xhw/桌面/Change-Detect/FarmlandD_data/FarmlandD_test/'
model = '/home/xhw/桌面/Change-Detect/checkpoints_D/CP10_val_0.011972.pth'
use_gpu = 1

net = Simple_CNN(n_channels=2, n_classes=1)
net.cuda()
net.load_state_dict(torch.load(model))

#初始化生成图像,（高，宽）
result_pic = np.zeros((289,257))
num = 0

#分组，每组个数为batch_size
test_img_list = os.listdir(test_img_path)
all_sub_list = []
batch_size = 64
for i in range(len(test_img_list)//batch_size):
    temp_list = test_img_list[i*batch_size:(i+1)*batch_size]
    all_sub_list.append(temp_list)
all_sub_list.append(test_img_list[:len(test_img_list)%batch_size])

for sub in all_sub_list:
    temp = []
    zuo_biao = []
    for sub_pic in sub:
        imgData = np.load(test_img_path+sub_pic)
        coord_x = int(sub_pic.split('_')[1])-2
        coord_y = int(sub_pic.split('_')[2][:-4])-2
        zuo_biao.append([coord_x, coord_y])
        imgData = np.expand_dims(imgData, 0)
        temp.append(imgData)
    sub_part = np.concatenate(temp, axis=0)
    sub_part = np.swapaxes(np.swapaxes(sub_part, 1, 3), 2, 3)
    sub_part = torch.from_numpy(sub_part).type(torch.FloatTensor)
    net.eval()
    if use_gpu:
        sub_part = sub_part.cuda()
    label_pred = net(sub_part).cpu().detach().numpy()
    label_pred = (label_pred>0.5)+0.
    for id, sub_pred in enumerate(label_pred):
        result_pic[zuo_biao[id][0],zuo_biao[id][1]] = sub_pred
    result = bilary2pic(result_pic)
    num += 1
    cv2.imwrite('/home/xhw/桌面/Change-Detect/checkpoints_D/FarmlandD_result10.png', result)
    print(num)





