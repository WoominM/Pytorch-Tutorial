import os
import sys

import torch 
import torch.nn as nn  
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchlight import DictAction

import numpy as np
import argparse
import random
import yaml
import csv
import traceback
import thop
from copy import deepcopy
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import confusion_matrix

import model
from CustomDataset import *


def init_seed(seed):
    torch.cuda.manual_seed_all(seed) # gpu固定
    torch.manual_seed(seed) # cpu固定
    np.random.seed(seed) # numpy固定
    random.seed(seed) # python固定
    torch.backends.cudnn.deterministic = True # 找出最优的卷积算法，保证复现性
    torch.backends.cudnn.benchmark = False # cudnn加速，网络结构固定时才有效

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
        
def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def get_parser():
    
    parser = argparse.ArgumentParser(
        description='Pytorch Tutorial')
    
    # directory
    parser.add_argument('--work-dir', 
                        default='./...',
                        help='the work folder for storing log')
    parser.add_argument('--save-dir', 
                        default='./...',
                        help='the work folder for storing results')
    parser.add_argument('--config', 
                        default='./config.yaml',
                        help='path to the configuration file')

    # train or test
    parser.add_argument('--phase', 
                        default='train', 
                        help='must be train or test')
    
    # feeder
    parser.add_argument('--num-worker',
                        type=int, default=8,
                        help='the number of worker for data loader')
    parser.add_argument('--use-mnist',
                        type=str2bool, default=True,
                        help='using MNIST Dataset for training or not')

    # model
    parser.add_argument('--model', 
                        default=None, 
                        help='the model will be used')
    parser.add_argument('--model-args',
                        action=DictAction, default=dict(),
                        help='the arguments of model')
    parser.add_argument('--weights',
                        default=None,
                        help='the weights for network initialization')

    # optim
    parser.add_argument('--lr', 
                        type=float, default=0.01, 
                        help='initial learning rate')
    parser.add_argument('--step',
                        type=int, default=[20, 40, 60], nargs='+',
                        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device',
                        type=int, default=0, nargs='+',
                        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', 
                        default='SGD', 
                        help='type of optimizer')
    parser.add_argument('--nesterov', 
                        type=str2bool, default=False, 
                        help='use nesterov or not')
    parser.add_argument('--batch-size', 
                        type=int, default=256, 
                        help='training batch size')
    parser.add_argument('--test-batch-size', 
                        type=int, default=256, 
                        help='test batch size')
    parser.add_argument('--start-epoch',
                        type=int, default=0,
                        help='start training from which epoch')
    parser.add_argument('--num-epoch',
                        type=int, default=80,
                        help='stop training in which epoch')
    parser.add_argument('--weight-decay',
                        type=float, default=0.0005,
                        help='weight decay for optimizer')
    parser.add_argument('--warm_up_epoch', 
                        type=int, default=0,
                        help='warm up strategy')
    
     # etc 
    parser.add_argument('--seed',
                        type=int, default=0, 
                        help='random seed for pytorch')
    parser.add_argument('--save-interval',
                        type=int, default=5,
                        help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch',
                        type=int, default=0,
                        help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-interval',
                        type=int, default=5,
                        help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log',
                        type=str2bool, default=True,
                        help='print logging or not')


    return parser

class Processor():
    def __init__(self, args):
        self.args = args
        self.load_data()
        self.load_model()
        self.load_optimizer()
        
        dataiter = iter(self.data_loader[self.args.phase])
        self.lr = self.args.lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        
        self.model = self.model.cuda(self.output_device)
        if type(self.args.device) is list:
            if len(self.args.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.args.device,
                    output_device=self.output_device)     

    def load_data(self):
        self.data_loader = dict()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        if self.args.phase == 'train':
            self.train_set = datasets.MNIST(root='./mnist', train=True, transform=transform, download=True) if self.args.use_mnist \
            else CustomDataset()
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=self.train_set,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers = self.args.num_worker,
                worker_init_fn=init_seed)

        self.test_set = datasets.MNIST(root='./mnist', train=False, transform=transform, download=True)  if self.args.use_mnist \
        else CustomDataset()  
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.test_set,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers = self.args.num_worker,
            worker_init_fn=init_seed)
    
    def load_model(self):
        output_device = self.args.device[0] if type(self.args.device) is list else self.args.device
        self.output_device = output_device
        model = import_class(self.args.model)
        self.model = model(**self.args.model_args)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        
        self.data_shape = [1,28,28] if self.args.use_mnist else self.train_set.getshape() 
        inputsample = torch.rand([1,1] + self.data_shape)
        self.flops, self.params = thop.profile(deepcopy(self.model), inputs=inputsample, verbose=False)
        
        if self.args.weights:
            weights = torch.load(self.args.weights)
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)
            
    def load_optimizer(self):
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()        
        
    def adjust_learning_rate(self, epoch):
        if self.args.optimizer == 'SGD' or self.args.optimizer == 'Adam':
            if epoch < self.args.warm_up_epoch:
                lr = self.args.lr * (epoch + 1) / self.args.warm_up_epoch
            else:
                lr = self.args.lr * (0.1 ** np.sum(epoch >= np.array(self.args.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()    
            
    def print_log(self, str, print_time=False):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.args.print_log:
            with open('{}/log.txt'.format(self.args.work_dir), 'a') as f:
                print(str, file=f)
                
    def train(self, epoch, save_model=True):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        train_loader = self.data_loader['train']
#         process = tqdm(train_loader, dynamic_ncols=True)
        process = tqdm(train_loader, ncols=100)
        self.adjust_learning_rate(epoch)
        
        loss_ = []
        acc_ = []
        for batch_idx, (data, label) in enumerate(process):
            with torch.no_grad():
                data, label = data.cuda(self.output_device), label.cuda(self.output_device)
            data, label = Variable(data), Variable(label)

            output = self.model(data)
            loss = self.loss(output, label)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()    
            
            loss_.append(loss.data.item())
            value, predicted_label = output.data.max(dim=1)
            acc = torch.mean((predicted_label == label.data).float())
            acc_.append(acc.data.item())
            
            self.lr = self.optimizer.param_groups[0]['lr']
            process.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.data.item(), self.lr))
        
        self.print_log('\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_), np.mean(acc_)*100))
        
        if save_model and epoch%self.args.save_interval==0:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.args.save_dir + '/epoch_' + str(epoch+1) + '.pt')
        
    
    def evaluate(self, epoch):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        test_loader = self.data_loader['test']
        loss_ = []
        acc_ = []
        label_list = []
        pred_list = []
#         process = tqdm(test_loader, dynamic_ncols=True)
        process = tqdm(test_loader, ncols=100)
        
        for batch_idx, (data, label) in enumerate(process):
            label_list.append(label)
            with torch.no_grad():
                data, label = data.cuda(self.output_device), label.cuda(self.output_device)
            data, label = Variable(data), Variable(label)
            
            output = self.model(data)
            loss = self.loss(output, label)
            
            loss_.append(loss.data.item())
            _, predicted_label = torch.max(output.data, 1)
            acc = torch.mean((predicted_label == label.data).float())
            acc_.append(acc.data.item())
            pred_list.append(predicted_label.data.cpu().numpy())
                        
        loss = np.mean(loss_)
        accuracy = np.mean(acc_)*100
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            self.best_acc_epoch = epoch + 1        
        self.print_log('\tMean test loss: {:.4f}.  Mean test acc: {:.2f}%.'.format(loss, accuracy))
        
        label_list = np.concatenate(label_list)
        pred_list = np.concatenate(pred_list)
        confusion = confusion_matrix(label_list, pred_list)
        list_diag = np.diag(confusion)
        list_raw_sum = np.sum(confusion, axis=1)
        each_acc = list_diag / list_raw_sum
        with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.args.save_dir, epoch + 1, ['test']), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(each_acc)
            writer.writerows(confusion)
            
        
    def start(self):
        if self.args.phase == 'train':
            self.print_log('Device: {}'.format(self.args.device if torch.cuda.is_available() else 'cpu')) 
            self.print_log('Model profile: {:.2f}G FLOPs and {:.2f}M Parameters'.format(self.flops / 1e9, self.params / 1e6))
            self.print_log('Pretrained weights: {}'.format('True' if self.args.weights else 'False'))
            self.print_log('Start epoch: {}'.format(self.args.start_epoch))
            
            self.print_log('\nStart Training...')
            
        
            for epoch in range(self.args.start_epoch, self.args.num_epoch):
                self.print_log('*'*100)
                self.train(epoch)
                if epoch % 5 == 0:
                    self.evaluate(epoch)
                self.print_log('Best_Accuracy: {:.2f}%, epoch: {}'.format(self.best_acc, self.best_acc_epoch))

            self.print_log('\nFinishi Training!')
            self.print_log('Best accuracy: {}'.format(self.best_acc))
            self.print_log('Epoch number: {}'.format(self.best_acc_epoch))
            self.print_log('Model name: {}'.format(self.args.work_dir))
            self.print_log('Model profile: {:.2f}G FLOPs and {:.2f}M Parameters'.format(self.flops / 1e9, self.params / 1e6))
            self.print_log('Weight decay: {}'.format(self.args.weight_decay))
            self.print_log('Base LR: {}'.format(self.args.lr))
            self.print_log('Batch Size: {}'.format(self.args.batch_size))
            self.print_log('Test Batch Size: {}'.format(self.args.test_batch_size))
            self.print_log('seed: {}'.format(self.args.seed))
        
        elif self.args.phase == 'test':
            if self.args.weights is None:
                raise ValueError('Please appoint --weights.')
            self.args.print_log = False
            self.print_log('Model:   {}.'.format(self.args.model))
            self.print_log('Weights: {}.'.format(self.args.weights))
            self.evaluate(epoch=0)
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()
    os.chdir(os.getcwd())
    p = parser.parse_args(args=[])
#     p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

#     arg = parser.parse_args()
    args = parser.parse_args(args=[])
#     args.phase = 'test'
#     args.weights = args.save_dir + '/epoch_16.pt'
#     args.work_dir = args.work_dir + '/'
#     args.start_epoch=100
    
    init_seed(args.seed)
    processor = Processor(args) 
    processor.start()