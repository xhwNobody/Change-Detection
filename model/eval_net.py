import os
from torch.utils.data import DataLoader
from loss import binary_loss
from load_data import SAR_Dataset

def eval_net(net, valImgPath, valLabPath, batch_size, gpu=True):
    val_img_list = os.listdir(valImgPath)
    net.eval()
    tot_loss = 0
    kits = SAR_Dataset(imgPath=valImgPath, labPath=valLabPath)
    test_loader = DataLoader(kits, batch_size, shuffle=False, num_workers=4)
    for i, data in enumerate(test_loader):
        imgs = data[0]
        true_labels = data[1]
        if gpu:
            imgs = imgs.cuda()
            true_labels = true_labels.cuda()
        mask_pred = net(imgs)
        tot_loss += binary_loss(mask_pred, true_labels).item()
    return tot_loss/ (len(val_img_list)//batch_size)