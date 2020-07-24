import cv2
import numpy as np

#计算相似性矩阵
def cal_similarity(pic1_path, pic2_path):
    pic1_data = cv2.imread(pic1_path)[...,0]
    pic2_data = cv2.imread(pic2_path)[...,0]
    pic_size = pic1_data.shape
    simility = np.zeros(pic_size)
    for i in range(pic_size[0]):
        for j in range(pic_size[1]):
            if int(pic1_data[i][j]) + int(pic2_data[i][j]) != 0:
                simility[i][j] = np.abs(int(pic1_data[i][j]) - int(pic2_data[i][j]))/(int(pic1_data[i][j]) + int(pic2_data[i][j]))
            else:
                simility[i][j] = 0
    return simility

#将0-1数值转化为二值化图像（可视化过程）
def bilary2pic(data):
    datasize = data.shape
    for i in range(datasize[0]):
        for j in range(datasize[1]):
            if data[i][j] == 1:
                data[i][j] = 255
    data = np.expand_dims(data, -1)
    result = np.concatenate([data, data, data], -1)
    return result

def FCM(X, c_clusters=2, m=2, eps=0.0001): #FCM算法
    X = np.array(X)
    membership_mat = np.random.random((len(X), c_clusters))
    membership_mat = np.divide(membership_mat, np.sum(membership_mat, axis=1)[:, np.newaxis])

    while True:
        working_membership_mat = membership_mat ** m
        Centroids = np.divide(np.dot(working_membership_mat.T, X),np.sum(working_membership_mat.T, axis=1)[:, np.newaxis])

        n_c_distance_mat = np.zeros((len(X), c_clusters))
        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                n_c_distance_mat[i][j] = np.linalg.norm(x - c, 2)

        new_membership_mat = np.zeros((len(X), c_clusters))

        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                new_membership_mat[i][j] = 1. / np.sum((n_c_distance_mat[i][j] / n_c_distance_mat[i]) ** (2 / (m - 1)))

        if np.sum(abs(new_membership_mat - membership_mat)) < eps:
            break
        membership_mat = new_membership_mat
    return np.argmax(new_membership_mat, axis=1)

def main():
    pic1_path = '.\\dataset\\FarmlandC\\200806.bmp'
    pic2_path = '.\\dataset\\FarmlandC\\200906.bmp'
    similarity = cal_similarity(pic1_path, pic2_path)
    result = FCM(np.reshape(similarity, (-1, 1)))

    result = np.reshape(result, similarity.shape)
    result = bilary2pic(result)
    # 必须为png，否则像素值会变化
    cv2.imwrite('.\\step1_Pseudo_label\\' + 'FarmlandC_pseudp_label.png', result)
main()