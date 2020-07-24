import cv2
import numpy as np

def evaluate(pred_label_data, true_label_data):
    TP, TN, FP, FN = 0, 0, 0, 0
    pred_label_data = np.reshape(pred_label_data, (1,-1))
    true_label_data = np.reshape(true_label_data, (1,-1))
    all_num = pred_label_data.shape[1]

    for i in range(all_num):
        if pred_label_data[0][i] == 255 and true_label_data[0][i] == 255:
            TP += 1
        elif pred_label_data[0][i] == 255 and true_label_data[0][i] == 0:
            FP += 1
        elif pred_label_data[0][i] == 0 and true_label_data[0][i] == 255:
            FN += 1
        else:
            TN += 1

    FPR = FP/(TP+FP+TN+FN)
    FNR = FN/(TP+FP+TN+FN)
    OE = FNR+FPR
    PCC = (TP+TN)/(TP+FP+TN+FN)
    PRE = ((TP+FP)*(TP+FN))/((TP+TN+FP+FN)**2) + ((FN+TN)*(FP+TN))/((TP+TN+FP+FN)**2)
    Kappa = (PCC-PRE)/(1-PRE)

    return FPR, FNR, OE, PCC, Kappa

def main():
    pred_label = '.\\0FarmlandD\\FarmlandD_result9.png'
    true_label = '.\\dataset\\FarmlandD\\reference.bmp'
    true_label_data = cv2.imread(true_label)[..., 0]
    pred_label_data = cv2.imread(pred_label)[..., 0]
    FP, FN, OE, PCC, Kappa = evaluate(pred_label_data, true_label_data)
    print('FP:' + str(FP))
    print('FN:' + str(FN))
    print('OE:' + str(OE))
    print('PCC:' + str(PCC))
    print('Kappa:' + str(Kappa))

main()




