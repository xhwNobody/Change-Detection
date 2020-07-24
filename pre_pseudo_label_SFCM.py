import random
import copy
import cv2
import numpy as np

try:
    import psyco

    psyco.full()
except ImportError:
    pass

FLOAT_MAX = 1e100

class Point:
    __slots__ = ["x", "group", "membership"]

    def __init__(self, clusterCenterNumber, x=0, group=0):
        self.x, self.group = x, group
        self.membership = [0.0 for _ in range(clusterCenterNumber)]

def similarityMatrix(imgA,imgB):
    similarity = np.absolute(imgA - imgB) / (imgA + imgB)  # numpy中的矩阵点除就是/，
    return similarity

def generatePoints(clusterCenterNumber, matrix, row, col):
    points = [Point(clusterCenterNumber) for _ in range(row * col)]
    matrix_flat = matrix.reshape((1,row*col))
    for index, point in enumerate(points):
        point.x = matrix_flat[0, index]
    return points

def solveDistanceBetweenPoints(pointA, pointB):
    return (pointA.x - pointB.x) * (pointA.x - pointB.x)

def getNearestCenter(point, clusterCenterGroup):
    minIndex = point.group
    minDistance = FLOAT_MAX
    for index, center in enumerate(clusterCenterGroup):
        distance = solveDistanceBetweenPoints(point, center)
        if (distance < minDistance):
            minDistance = distance
            minIndex = index
    return (minIndex, minDistance)

def kMeansPlusPlus(points, clusterCenterGroup):
    clusterCenterGroup[0] = copy.copy(random.choice(points))  # 在所有的points里面随机选一个点
    distanceGroup = [0.0 for _ in range(len(points))]  # 把群心到所有points的距离矩阵初始化为零矩阵
    sum = 0.0
    for index in range(1, len(clusterCenterGroup)):  # 索引各个群心点
        for i, point in enumerate(points):  # Python中的Enumerate().很多时候，在处理迭代器时，我们还需要保留迭代次数.
            distanceGroup[i] = getNearestCenter(point, clusterCenterGroup[:index])[1]  # 对一个群心点，计算所有点到该群心点的距离平方
            sum += distanceGroup[i]  #
        sum *= random.random()
        for i, distance in enumerate(distanceGroup):
            sum -= distance
            if sum < 0:
                clusterCenterGroup[index] = copy.copy(points[i])
                break
    return

def fuzzyCMeansClustering(points, clusterCenterNumber, weight):
    clusterCenterGroup = [Point(clusterCenterNumber) for _ in range(clusterCenterNumber)]
    kMeansPlusPlus(points, clusterCenterGroup)
    clusterCenterTrace = [clusterCenter for clusterCenter in clusterCenterGroup]
    tolerableError, currentError = 1.0, FLOAT_MAX
    while currentError >= tolerableError:
        for point in points:
            getSingleMembership(point, clusterCenterGroup, weight)  # 根据群心确定权重系数U_ij,这个操作在循环体内，会随着群心的更新而更新
        currentCenterGroup = [Point(clusterCenterNumber) for _ in range(clusterCenterNumber)]
        for centerIndex, center in enumerate(currentCenterGroup):
            upperSumX, lowerSum = 0.0, 0.0
            for point in points:
                membershipWeight = pow(point.membership[centerIndex], weight)  # U_ij的m次方，权重系数
                upperSumX += point.x * membershipWeight  # 横坐标和纵坐标分别乘以权重
                lowerSum += membershipWeight  # 计算Ci的分母
            center.x = upperSumX / lowerSum  # 更新群心
            print(center.x)
        # update cluster center trace
        currentError = 0.0
        for index, singleTrace in enumerate(clusterCenterTrace):  # index迭代次数累加，singleTrace群心的坐标
            # singleTrace.append(currentCenterGroup[index])  # singleTrace
            currentError += solveDistanceBetweenPoints(singleTrace, currentCenterGroup[index])
            # currentError += solveDistanceBetweenPoints(singleTrace[-1], singleTrace[-2])
            clusterCenterGroup[index] = copy.copy(currentCenterGroup[index])  # 做更新
    # 从循环体出来，目标函数已经收敛到理想区间了
    for point in points:
        maxIndex, maxMembership = 0, 0.0
        for index, singleMembership in enumerate(point.membership):
            if singleMembership > maxMembership:
                maxMembership = singleMembership
                maxIndex = index
        point.group = maxIndex
    return clusterCenterGroup, clusterCenterTrace

def getSingleMembership(point, clusterCenterGroup, weight):  # 公式1
    distanceFromPoint2ClusterCenterGroup = [solveDistanceBetweenPoints(point, clusterCenterGroup[index]) for index in
                                            range(len(clusterCenterGroup))]  # 某点到各个群心的距离矩阵size = (1,k)
    for centerIndex, singleMembership in enumerate(point.membership):  # 该点到各个群心的概率矩阵
        sum = 0.0
        isCoincide = [False, 0]
        for index, distance in enumerate(distanceFromPoint2ClusterCenterGroup):
            if distance == 0:
                isCoincide[0] = True  # 如果群心刚好是点集里的某个点，那么就是isCoincide
                isCoincide[1] = index
                break
            # 如果distance不为零
            sum += pow(float(distanceFromPoint2ClusterCenterGroup[centerIndex] / distance),
                       1.0 / (weight - 1.0))  # 公式1的分母
        if isCoincide[0]:
            if isCoincide[1] == centerIndex:
                point.membership[centerIndex] = 1.0
            else:
                point.membership[centerIndex] = 0.0
        else:
            point.membership[centerIndex] = 1.0 / sum

def ClusterAnalysisResults(points,row,col):
    Binary_matrix = np.zeros((1, row * col), dtype='int')
    for index,point in enumerate(points):
        Binary_matrix[0,index] = point.group
    Binary_matrix = Binary_matrix.reshape(row,col)
    return Binary_matrix

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

def bilary2pic(data): #将0-1数值转化为二值化图像（可视化过程）
    datasize = data.shape
    for i in range(datasize[0]):
        for j in range(datasize[1]):
            if data[i][j] == 1:
                data[i][j] = 255
    data = np.expand_dims(data, -1)
    result = np.concatenate([data, data, data], -1)
    return result

def main():

    pic1_path = '.\\dataset\\FarmlandD\\200806.bmp'
    pic2_path = '.\\dataset\\FarmlandD\\200906.bmp'
    A = cal_similarity(pic1_path, pic2_path)
    [row, col] = A.shape
    clusterCenterNumber = 2
    weight = 2
    points = generatePoints(clusterCenterNumber,A,row,col)
    _, clusterCenterTrace = fuzzyCMeansClustering(points, clusterCenterNumber, weight)
    result = ClusterAnalysisResults(points, row,col)
    result_pic = bilary2pic(result)
    # 必须为png，否则像素值会变化
    cv2.imwrite('.\\step1_Pseudo_label\\' + 'FarmlandD_pseudp_label.png', result_pic)

main()