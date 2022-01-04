
import argparse
import utils.losses as losses
from utils.utils import str2bool

def parse_args():
    loss_names = list(losses.__dict__.keys())
    loss_names.append('BCEWithLogitsLoss')

    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', '-a', metavar='ARCH', default='Unet') #Unet AttU_Net
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default='STS',
                        help='dataset name')
    parser.add_argument('--input-channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--aug', default=True, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',  #100
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)') #60
    parser.add_argument('-b', '--batch-size', default=10, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, #1e-4
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=5e-5, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--original_image_path', default='/home/qiaoxiaoya/workspace/code/Public_Norm/public_dataset/STS/', type=str, help='original train image path')
    parser.add_argument('--struct_name', default='AT', type=str, help='structure name')

    args = parser.parse_args()

    return args
