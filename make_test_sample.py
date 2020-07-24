import cv2
import numpy as np

def make_test_sample(pic_time1, pic_time2, dst_image_path):
    pic_time1_data = cv2.imread(pic_time1)[...,0]
    pic_time2_data = cv2.imread(pic_time2)[...,0]
    print(pic_time1_data.shape)

    pic_time1_data_pad = np.pad(pic_time1_data, 2) #padding width+4,height+4
    pic_time2_data_pad = np.pad(pic_time2_data, 2) #padding width+4,height+4
    print(pic_time1_data_pad.shape)

    data_size = pic_time1_data_pad.shape
    patch_size = 5
    for i in range(0, data_size[0]-patch_size+1):
        for j in range(0, data_size[1]-patch_size+1):
            pic1_patch = pic_time1_data_pad[i:i+patch_size, j:j+patch_size]
            pic1_path_pad = np.expand_dims(np.pad(pic1_patch, 1), -1)
            pic2_patch = pic_time2_data_pad[i:i+patch_size, j:j+patch_size]
            pic2_path_pad = np.expand_dims(np.pad(pic2_patch, 1), -1)
            merge_patch = np.concatenate([pic1_path_pad, pic2_path_pad], axis=-1)
            np.save(dst_image_path + pic_time1.split('\\')[-2] + '_'+ str(i+(patch_size//2)) + '_' +str(j+(patch_size//2)) + '.npy', merge_patch)

def main():
    pic_time1 = '.\\dataset\\FarmlandC\\200806.bmp'
    pic_time2 = '.\\dataset\\FarmlandC\\200906.bmp'
    dst_image_path = '.\\step3_test_dataset\\FarmlandC_test\\'
    make_test_sample(pic_time1, pic_time2, dst_image_path)

main()
