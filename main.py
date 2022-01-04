# -*- coding: utf-8 -*-
# xiaoya's demo

import sys
import os
import joblib

import torch
import torch.backends.cudnn as cudnn

from data_loader import DataLoaderGenerator
from parse_args import parse_args
from model_train import ModelTrainer
from Unet.unet_runtime_generator import UnetRuntimeGenerator

os.environ["CUDA_VISIBLE_DEVICES"]= '0'
print('GPU Device Number is %d '%(torch.cuda.current_device()))
CONTINUE_TRAIN = False

def main(args):

    if not os.path.exists('models/%s' %args.struct_name):
        os.makedirs('models/%s' %args.struct_name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')
    with open('models/%s/args.txt' %args.struct_name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.struct_name)

    # accelarate or not
    cudnn.benchmark = True
    print("=> creating model %s" %args.struct_name)

    model = UnetRuntimeGenerator.generateUnet(args.struct_name)
    if CONTINUE_TRAIN: 
        model.load_state_dict(torch.load('models/%s/model.pth' %args.struct_name))
    model = model.cuda()
    
    data_loader_generator = DataLoaderGenerator(args)
    train_loader, val_loader = data_loader_generator.generate()

    model_trainer = ModelTrainer(args, train_loader, val_loader, model)
    model_trainer.train_model()
    model_trainer.tidy_up()

if __name__ == '__main__':
    args = parse_args()
    main(args)
