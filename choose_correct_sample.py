import cv2
import os
import numpy as np

# 将图像二值化
def pic2binary(data):
    datasize = data.shape
    for i in range(datasize[0]):
        for j in range(datasize[1]):
            if data[i][j] == 255:
                data[i][j] = 1
    return data

def choose_correct_sample(pic_time1, pic_time2, pseudo_label, dst_image_path, dst_label_path):
    pic_time1_data = cv2.imread(pic_time1)[...,0]
    pic_time2_data = cv2.imread(pic_time2)[...,0]
    label_data = cv2.imread(pseudo_label)[...,0]
    label_data = pic2binary(label_data)
    data_size = label_data.shape
    patch_size = 5
    for i in range(0, data_size[0]-patch_size):
        for j in range(0, data_size[1]-patch_size):
            patch_data = label_data[i:i+patch_size, j:j+patch_size] #5*5
            center_label = label_data[i+(patch_size//2) ,j+(patch_size//2)] #取中心点的值
            if np.sum((patch_data == center_label) + 0.)/(patch_size**2) > 0.6: #公式判断
                pic1_patch = pic_time1_data[i:i+patch_size, j:j+patch_size]
                pic1_path_pad = np.expand_dims(np.pad(pic1_patch, 1), -1) #padding 7*7
                pic2_patch = pic_time2_data[i:i+patch_size, j:j+patch_size]
                pic2_path_pad = np.expand_dims(np.pad(pic2_patch, 1), -1) #padding 7*7
                merge_patch = np.concatenate([pic1_path_pad, pic2_path_pad], axis=-1) #拼接

                np.save(dst_image_path + pic_time1.split('\\')[-2] + '_' + str(i+(patch_size//2)) + '_' +str(j+(patch_size//2)) + '.npy', merge_patch)
                np.save(dst_label_path + pic_time1.split('\\')[-2] + '_' + str(i+(patch_size//2)) + '_' + str(j+(patch_size//2)) + '.npy', center_label)

def main():
    # 原始路径
    pic_time1 = '.\\dataset\\Ottawa\\199707.png'
    pic_time2 = '.\\dataset\\Ottawa\\199708.png'
    pseudo_label = '.\\step1_Pseudo_label\\Ottawa_pseudp_label_end.png'

    # 目标路径
    dst_image_path = '.\\step2_dataset_dst\\Ottawa_end\\image\\'
    dst_label_path = '.\\step2_dataset_dst\\Ottawa_end\\label\\'

    if not os.path.exists(dst_image_path):
        os.mkdir(dst_image_path)
    if not os.path.exists(dst_label_path):
        os.mkdir(dst_label_path)

    choose_correct_sample(pic_time1, pic_time2, pseudo_label, dst_image_path, dst_label_path)

main()
