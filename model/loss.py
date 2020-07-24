import torch.nn.functional as F

#二元交叉熵损失
def binary_loss(input, target):

    return F.binary_cross_entropy(input, target)