import os
import sys
import torch
from loss import binary_loss
from torch import optim
from optparse import OptionParser
from build_model import Simple_CNN
from load_data import SAR_Dataset
from torch.utils.data import DataLoader
from eval_net import eval_net

def train_net(net,
              epochs=5,
              batch_size=2,
              lr=0.1,
              save_cp=True,
              gpu=True):
    train_img_dir = './FarmlandC_data/train/image/'
    train_msk_dir = './FarmlandC_data/train/label/'
    val_img_dir = './FarmlandC_data/val/image/'
    val_msk_dir = './FarmlandC_data/val/label/'
    dir_checkpoint = './checkpoints_C/'

    train_img_list = os.listdir(train_img_dir)
    val_img_list = os.listdir(val_img_dir)

    print('''
Starting training:
    Training size: {}
    Validation size: {}
    Epochs: {}
    Batch size: {}
    Learning rate: {}
    Checkpoints: {}
    CUDA: {}
    '''.format(len(train_img_list), len(val_img_list), epochs, batch_size, lr, str(save_cp), str(gpu)))

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    max_val_loss = 1000.
    iter_num = 0
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch, epochs))
        net.train()
        sub_num = 0
        epoch_loss = 0
        kits = SAR_Dataset(imgPath=train_img_dir, labPath=train_msk_dir)
        train_loader = DataLoader(kits, batch_size, shuffle=True, num_workers=4)

        for i, data in enumerate(train_loader):
            imgs = data[0]
            true_masks = data[1]
            if gpu:
                imgs = imgs.float().cuda()
                true_masks = true_masks.float().cuda()
            masks_pred = net(imgs)
            loss = binary_loss(masks_pred, true_masks)
            print('iter:' + str(sub_num) + ' ' + str('%4f' % loss.item()))
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sub_num += 1

        #训练时进行验证
        with torch.no_grad():
            val_loss = eval_net(net, val_img_dir, val_msk_dir, batch_size,  gpu)
            print('Validation loss: {}'.format(val_loss))

        #如果验证损失变小，就保存模型
        if save_cp and val_loss<max_val_loss:
            torch.save(net.state_dict(), dir_checkpoint + 'CP{}_val_{}.pth'.format(iter_num, '%4f' % val_loss))
            print('Checkpoint {} saved !'.format(iter_num))
            max_val_loss = val_loss
        iter_num += 1

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=10, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=32,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.0001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = Simple_CNN(n_channels=2, n_classes=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
