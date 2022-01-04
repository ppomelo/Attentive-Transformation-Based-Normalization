import torch.nn as nn
from Unet.unet_struct import Unet
from Unet.unet_norm_all import AttentiveNorm2d,AttentiveTrans2d,LocalContextNorm,InstanceEnhancementBatchNorm2d

def group_norm_alias(out_ch):
    return nn.GroupNorm(4, out_ch)

class UnetRuntimeGenerator:

    name_to_norm_func_map = {
        "AN": [AttentiveNorm2d, AttentiveNorm2d],
        "AT": [AttentiveTrans2d, AttentiveTrans2d],
        "LCN":[LocalContextNorm, LocalContextNorm],
        "IEBN":[InstanceEnhancementBatchNorm2d, InstanceEnhancementBatchNorm2d],
        "IN": [nn.InstanceNorm2d, nn.InstanceNorm2d],
        "BN":[nn.BatchNorm2d, nn.BatchNorm2d],
        "GN":[group_norm_alias, group_norm_alias],
        "NONE":[None,None]
    }

    @staticmethod
    def generateUnet(name):
        norm_func_list = UnetRuntimeGenerator.name_to_norm_func_map[name]
        return Unet(init_func, norm_func_list[0], norm_func_list[1])


def init_func(in_ch, out_ch, norm_func1, norm_func2, factor1=True, factor2=True):
    if factor1 and factor2:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            norm_func1(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            norm_func2(out_ch),
            nn.ReLU(inplace=True)
        )
    if factor1:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            norm_func1(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    if factor2:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            norm_func2(out_ch),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

