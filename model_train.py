import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from utils.average_calculator import AverageCalculator
from collections import OrderedDict
from utils.losses import BCEDiceLoss
from utils.evaluation import dice_coef_hard
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

class ModelTrainer():
    def __init__(self, main_args, train_loader, val_loader, model):
        # magic nums
        self.save_step = 10
        
        self.main_args = main_args
        self.writer = SummaryWriter(comment=main_args.struct_name)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        # init optimizer and scheduler
        if main_args.optimizer == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = main_args.lr,
                amsgrad = main_args.nesterov)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=10,gamma=0.6)
        else: # indeed it only includes 'SGD'
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr = main_args.lr,
                momentum = main_args.momentum, weight_decay = main_args.weight_decay, nesterov = main_args.nesterov)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=10,gamma=0.6)
        
        # define loss function (criterion)
        if self.main_args.loss == 'BCEWithLogitsLoss':
            self.criterion = nn.BCEWithLogitsLoss().cuda()
        else :
            self.criterion = BCEDiceLoss().cuda()

    
    def do_train(self, epoch):

        losses = AverageCalculator()
        dices = AverageCalculator()
        
        self.model.train()

        for i, (input, target) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if self.main_args.deepsupervision:
                outputs = self.model(input)
                loss = 0
                for output in outputs:
                    loss += self.criterion(output, target)
                loss /= len(outputs)
            else:
                output = self.model(input,target)
                loss = self.criterion(output, target)
                dicecoef = dice_coef_hard(output,target)
                if i%self.save_step==0:
                    number_count_train = (i+(epoch-1)*len(self.train_loader))//self.save_step
                    self.writer.add_scalar('loss/train',loss.item(),number_count_train)
                    self.writer.add_scalar('dicecoef/train',dicecoef.item(),number_count_train)

            losses.update(loss.item(), input.size(0))
            dices.update(dicecoef,input.size(0))

            # compute gradient and do optimizing step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            nn.utils.clip_grad_value_(self.model.parameters(),0.1)

        print(self.optimizer.state_dict()['param_groups'][0]['lr'])

            # for tag, value in self.model.named_parameters():
            #     tag = tag.replace('.','/')
            #     self.writer.add_histogram('weights/'+tag,value.data.cpu().numpy(),i)
            #     self.writer.add_histogram('grads/'+tag,value.grad.data.cpu().numpy(),i)

        return OrderedDict([
            ('loss', losses.avg),
            ('dice',dices.avg)
        ])

    def do_validate(self, epoch):
        losses = AverageCalculator()
        dices = AverageCalculator()


        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            for i, (input, target) in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                input = input.cuda()
                target = target.cuda()

                # compute output
                if self.main_args.deepsupervision:
                    outputs = self.model(input,target)
                    loss = 0
                    for output in outputs:
                        loss += self.criterion(output, target)
                    loss /= len(outputs)
                else:
                    output = self.model(input,target)
                    loss = self.criterion(output, target)
                    dicecoef = dice_coef_hard(output,target)
                    if i%self.save_step == 0:
                        number_count_val = (i+(epoch-1)*len(self.val_loader))//self.save_step
                        self.writer.add_scalar('loss/valid',loss.item(),number_count_val) #del ex-test
                        self.writer.add_scalar('dicecoef/valid',dicecoef.item(),number_count_val)
                    
                    
                losses.update(loss.item(), input.size(0))
                dices.update(dicecoef.item(),input.size(0))

        return OrderedDict([
            ('loss', losses.avg),
            ('dice',dices.avg)
        ])

    def train_model(self):

        # Data structure definition
        log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss','dice', 'val_loss', 'val_dice'])

        # Iteration init logic
        best_dice = 0
        trigger = 0

        # Iteration starts
        for epoch in range(1, self.main_args.epochs):
            print('Epoch [%d/%d]' %(epoch, self.main_args.epochs))

            train_log = self.do_train(epoch)
            self.scheduler.step()
            val_log = self.do_validate(epoch)

            train_loss = train_log['loss']
            train_dice = train_log['dice']
            
            val_loss = val_log['loss']
            val_dice = val_log['dice']

            print('loss %.4f - dice %.4f - val_loss %.4f -val_dice %.4f'
                %(train_loss, train_dice, val_loss, val_dice))

            temp_series = pd.Series([
                epoch,
                self.optimizer.state_dict()['param_groups'][0]['lr'],
                train_loss,
                train_dice,
                val_loss,
                val_dice
            ], index=['epoch', 'lr', 'loss', 'dice', 'val_loss', 'val_dice'])

            log = log.append(temp_series, ignore_index=True)
            log.to_csv('models/%s/log.csv' %self.main_args.struct_name, index=False)

            # save best model         
            trigger += 1

            if val_log['dice'] > best_dice:
                torch.save(self.model.state_dict(), 'models/%s/model.pth' %self.main_args.struct_name)
                best_dice = val_log['dice']
                print("=> saved best model")
                trigger = 0

            # early stopping
            if self.main_args.early_stop:
                if trigger >= self.main_args.early_stop:
                    print("=> early stopping")
                    break

            torch.cuda.empty_cache()

    def tidy_up(self):
        
        self.writer.close()