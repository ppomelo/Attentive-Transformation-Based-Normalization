import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk


def batch_iou(output, target):

    output = torch.sigmoid(output).data.cpu().numpy() > 0.5
    target = (target.data.cpu().numpy() > 0.5).astype('int')
    output = output[:,0,:,:]
    target = target[:,0,:,:]

    ious = []
    for i in range(output.shape[0]):
        ious.append(mean_iou(output[i], target[i]))

    return np.mean(ious)


def mean_iou(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    ious = []
    for t in np.arange(0.5, 1.0, 0.1):
        output_ = output > t
        target_ = target > t
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return np.mean(ious)


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def dice_coef_soft(input, target):
    
    smooth = 1e-5
    input = torch.sigmoid(input)
    num = target.size(0)
    input = input.view(num, -1).data.cpu().numpy()
    target = target.view(num, -1).data.cpu().numpy()
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    return dice.sum()/num

def dice_coef_hard(input, target):

    smooth = 1e-5
    input = torch.sigmoid(input)
    input = (input >= 0.5).float()
    num = target.size(0)
    input = input.view(num, -1).data.cpu().numpy()
    target = target.view(num, -1).data.cpu().numpy()
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    return dice.sum()/num


def accuracy(output, target):
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output = (np.round(output)).astype('int')
    target = target.view(-1).data.cpu().numpy()
    target = (np.round(target)).astype('int')
    (output == target).sum()
    return (output == target).sum() / len(output)

def ppv(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
        output = np.round(output)
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
        target = np.round(target)
    intersection = (output * target).sum()
    return  (intersection + smooth) / \
           (output.sum() + smooth)

def sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
        output = np.round(output)
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
        # target = np.round(target)

    intersection = (output * target).sum()

    return (intersection + smooth) / \
        (target.sum() + smooth)

def hausdorff(output,target):

    if torch.is_tensor(output):
        output = torch.sigmoid(output)
        output = output.data.cpu().numpy().squeeze()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy().squeeze()
    labelPred=sitk.GetImageFromArray(output, isVector=False)
    labelTrue=sitk.GetImageFromArray(target, isVector=False)
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>=0.5,labelPred>=0.5)
    haus=hausdorffcomputer.GetHausdorffDistance()

    return haus
