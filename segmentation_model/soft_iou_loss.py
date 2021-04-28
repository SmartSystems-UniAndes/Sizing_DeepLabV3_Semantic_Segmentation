import torch
import torch.nn.functional as F

from torch import nn


class SoftIOULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2, device=None):
        super(SoftIOULoss, self).__init__()
        self.classes = n_classes
        self.device = device

    def forward(self, inputs, target):
        # inputs n x c x h x w
        n, n_classes, h, w = inputs.size()
        # target n x c x h x w
        target = torch.zeros(n, n_classes, h, w).to(self.device).scatter_(1, target.view(n, 1, h, w), 1)

        inputs = F.softmax(inputs, dim=1)

        intersection = inputs * target
        # n x c x h x w => n x c
        intersection = intersection.view(n, self.classes, -1).sum(2)

        union = inputs + target - (inputs*target)
        # n x c x h x w => n x c
        union = union.view(n, self.classes, -1).sum(2)

        iou = intersection/union

        loss = 1 - iou.mean()

        return loss
