# **Introduction**  <br>
**< Pytorch Tutorial >**<br><br>

### Prerequisites
+ python
+ pytorch
+ pyyaml/tqdm/... (installation)
+ Run ```pip install -e torchlight```
    <br><br>
    
### Dataset
+ Default: MNIST
+ if you want to use ```CustomDataset.py``` , set ```use_mnist=False``` in config file or ```python main.py use-mnist False``` 
    <br><br>

### Training & Testing
+ Change the file ```config.py``` on what you want
+ train : ```python main.py```<br>
pretrained model: ```python main.py --weights <save_dir>/xxx.pt```
+ test : ```python main.py --phase test --weights <save_dir>/xxx.pt```

    <br>  

# **Installation**  <br>
---

```python
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
```
torch 相关包解释：<br>
<br>
+ ```torch.nn```: 里面包含各种卷积,loss，激活函数
+ ```torchvision```: 用来处理图像的库。常用的子包：   
    ```.datasets```: 里面包含经典的图像数据集（如 MNIST、CIFAR、ImageNet等）   
    ```.models```: 里面包含经典的预训练模型（如 VGG、ResNet、DenseNet等）   
    ```.transforms```: 各种图像转换操作（如 转换为Tensor或Crop等）  
+ ```torchlight```: 用来处理parser中的dict类型数据  

# **Prepare Dataset**  <br>
---
```python
class CustomDataset(torch.utils.data.Dataset): 
    def __init__(self):
        bs = 128
        self.x_data = torch.rand([bs,1,28,28])
        self.y_data = torch.randint(0,10,[bs])
    
    def __len__(self): 
        return len(self.x_data)
    
    def __getitem__(self, idx): 
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y
    
    def getshape(self):
        size = self.x_data[0].size()
        return list(size)
```
```torchvision.dataset```里已有常用的图像数据集，本代码将采用其中的MNIST数据集。<br>
若需要用自己的Dataset进行训练，把```class```定义为 ```torch.utils.data.Dataset``` 的子类。  
<br>
里面需要设置：
+ __ init __ : 预定义，形成/提取数据
+ __ len __ : 返回样本数
+ __ getitem __ : 返回idx对应的样本  

另外还可以自行定义别的函数，如```topk```或```getshape```等。其中```getshape```是用来计算模型的参数量以及FLOPs。
    <br> <br> 
    
提取时需要定义提取器 ```torch.utils.data.DataLoader```。
<br><br>
_Example)_
```python
dataset = CustomDataset()
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)
for batch_idx, (x, y) in enumerate(data_loader):
    ...
```

# **Model**  <br>
---

### Initialization
```python
def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
```
```torch.nn.init```里面包含常用的初始化方案，如:<br>
+ 均匀分布：```torch.nn.init.uniform_(tensor, a, b)```
+ 正态分布：```torch.nn.init.normal_(tensor, mean, std)```
+ 常数：```torch.nn.init.constant_(tensor, val)```
+ Xavier/Kaiming：```uniform_```或```normal_```前面加```xavier_```或```kaiming_```即可   
<br><br>

### Module
```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ResBlock, self).__init__()
        if mid_channels == None:
            mid_channels = in_channels//4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels))
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else (lambda x:x)
        self.relu = nn.ReLU() 
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x).relu()
        x = x + self.down(res)
        return x
```
首先需要把```class```定义为```nn.Module```的子类。<br><br>

细节：<br>
1. 定义各 _Layer_ ：<br>
    以2d为例，输入形式：(N, C, H, W) <br>
    + 卷积：```nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)```  
    + BatchNorm：```nn.BatchNorm2d(channels)```  
    + 激活函数：```nn.ReLU()```，```nn.Sigmoid()```，```nn.Softmax(dim)```等   
        <br><br>  
    
2. 多个Layer合并方法常用的有3种：
    + ```nn.Sequential(layer1, layer2, ...)```：需要合并的层数交少的时候使用。
    + ```nn.ModuleList([layer1, layer2, ...])```：需要合并的层数交多的时候使用，如 ```nn.ModuleList([layer1 for i in range(10)])```
    + ```nn.Sequential```：先把```class```定义为 ```nn.Sequential```的子类，然后```init```里面通过```self.add_module('name', layer)```添加所需要的层，此方法比较灵活，条件加层时可以使用。  
        _Example)_

        ```python
        class Blocks(nn.Sequential):
            def __init__(self, block_args):
                super(Blocks, self).__init__()
                for i, channels in enumerate(block_args):
                    if len(channels) != 2:
                        in_c, mid_c, out_c = channels
                        self.add_module(f'resblock-{i}', ResBlock(in_c, out_c, mid_channels=mid_c))
                    else:
                        in_c, out_c = channels
                        self.add_module(f'resblock-{i}', ResBlock(in_c, out_c))
        ```
        <br>

3. 常用运算：
    + 加法：```torch.sum(tensor, dim)``` or ```tensor.sum(dim)```
    + 平均：```torch.mean(tensor, dim)``` or ```tensor.mean(dim)```
    + 乘法：```torch.matmul(a, b)``` or ```torch.einsum('nchw,nchu->ncwu', [a,b])```。推荐使用```torch.einsum()```，用起来很方便，并且基本上可以做出所有乘法。
    + 变形：```x.reshape(*shape)``` or ```x.view(*shape)```。两者结果相同，但注意使用```view```时需要保证```tensor```是连续的，否则报错，此时要：```x.contiguous().view(*shape)```。
    + 维度换位：```x.permute(*dimlist)```。注意```permute```之后```tensor```将变成不连续，因此一般与```contiguous()```一起使用，如：```x.permute(0,1,3,2).contiguous()```。
    + 合并：```torch.cat([x1,x2,...], dim)```
    + 分割：```torch.split(x, [n1,n2,...], dim)``` or ```torch.chunk(x, n, dim)```
    + 激活函数：如果激活函数里没有可学习参数的话，可以直接做 ```x.relu()```，而不需要```init```里面定义```nn.ReLU()```之后再使用```self.relu(x)```。
    + 滑动窗口：```nn.unfold(window_size, dilation, stride, padding)```，将输出(N, window_size*C, HW)。

## **Random Seed**  <br>
---
```python
def init_seed(seed):
    torch.cuda.manual_seed_all(seed) # gpu固定
    torch.manual_seed(seed) # cpu固定
    np.random.seed(seed) # numpy固定
    random.seed(seed) # python固定
    torch.backends.cudnn.deterministic = True # 找出最优的卷积算法，保证复现性
    torch.backends.cudnn.benchmark = False # cudnn加速，网络结构固定时才有效
```
训练之前固定随机种子，这样才能保证在同一机器上的复现性，但训练时间会变慢。随着不同pytorch版本，它的优化结果也会发生变化。

# **Paser**  <br>
---
```python
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
```
各参数解释可见help

## **Functional**  <br>
---
```python
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
```
功能函数<br>
+ str2bool: 输出True or False。   
+ import_class: 加载模型, 可见processor模块的load_model函数

# **Process**  <br>
---
processor模块功能：加载数据和模型，训练，测试。  
    <br>
```python
class Processor():
    def __init__(self, args):
        self.args = args #加载参数
        self.load_data() #加载数据
        self.load_model() #加载模型
        self.load_optimizer() #加载优化器

        dataiter = iter(self.data_loader[self.args.phase]) #提取数据集
        self.lr = self.args.lr #学习率
        self.best_acc = 0 #最佳准确率
        self.best_acc_epoch = 0 #最佳准确率对应的epoch

        self.model = self.model.cuda(self.output_device)  #多GPU
        if type(self.args.device) is list:
            if len(self.args.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.args.device,
                    output_device=self.output_device) 
```
<br>
子函数: <br>

+ load_data(): 加载数据  
    ```python
    def load_data(self):
        self.data_loader = dict()
        transform = transforms.Compose([ #数据变换
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        if self.args.phase == 'train':
            self.train_set = datasets.MNIST(root='./mnist', train=True, transform=transform, download=True) if self.args.use_mnist else CustomDataset()
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=self.train_set,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers = self.args.num_worker,
                worker_init_fn=init_seed)

        self.test_set = datasets.MNIST(root='./mnist', train=False, transform=transform, download=True) if self.args.use_mnist else CustomDataset()  
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.test_set,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers = self.args.num_worker,
            worker_init_fn=init_seed)
    ```   
   dataset可以使用```torchvision```里提供的数据集(本代码采用 MNIST)，也可以自行定义Dataset（见上面```Prepare Dataset```）。  
   训练时通过```enumerate(data_loader)```来读取数据，它将返回```batch_size```个```__getitem__()```的返回值。<br><br>  
   
+ load_model(): 加载模型  
    ```python
    def load_model(self):
        output_device = self.args.device[0] if type(self.args.device) is list else self.args.device
        self.output_device = output_device
        model = import_class(self.args.model) #加载模型
        self.model = model(**self.args.model_args)
        self.loss = nn.CrossEntropyLoss().cuda(output_device) #定义loss函数
        
        self.data_shape = [1,28,28] if self.args.use_mnist else self.train_set.getshape() 
        inputsample = torch.rand([1,1] + self.data_shape)
        self.flops, self.params = thop.profile(deepcopy(self.model), inputs=inputsample, verbose=False)  #计算参数量和FLOPs
        
        if self.args.weights: #加载预训练模型
            weights = torch.load(self.args.weights)
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)
    ```
    加载```model.py```里定义的模型。  
    使用```thop.profile(model, (inputshape,))```可以计算模型的参数量以及FLOPs   
    若需要加载预训练模型，先用```torch.load()```加载预训练权重，再通过```.load_state_dict(weights)```复制到我们的模型，代码中对应```if args.weights:```。<br><br>

+ load_optimizer(): 加载优化器 
    ```python
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
    ```
    ```torch.optim```里可以加载SGD或Adam等常用的优化器。<br><br>
   
+ adjust_learning_rate(): 学习策略 
    ```python
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
    ```
    可以使用```torch.optim```里提供的常用Scheduler(如```torch.optim.StepLR```)，若需要使用其他学习策略(如本代码中的 warm up strategy)，也可以自行定义Scheduler。<br><br>  
   
+ train(): 训练模型
    ```python
    def train(self, epoch, save_model=True):
        self.model.train() #训练模式
        self.print_log('Training epoch: {}'.format(epoch + 1))
        train_loader = self.data_loader['train'] #加载数据提取器
        process = tqdm(train_loader, ncols=100) #可视化工具
        self.adjust_learning_rate(epoch) #Scheduler
        
        loss_ = []
        acc_ = []
        for batch_idx, (data, label) in enumerate(process): #提取数据
            with torch.no_grad():
                data, label = data.cuda(self.output_device), label.cuda(self.output_device)
            data, label = Variable(data), Variable(label) 

            output = self.model(data) #模型输出
            loss = self.loss(output, label) #计算loss
            
            self.optimizer.zero_grad() #梯度初始化为0
            loss.backward() #反向传播得到梯度
            self.optimizer.step() #通过梯度下降法执行参数更新    
            
            loss_.append(loss.data.item()) 
            value, predicted_label = output.data.max(dim=1)
            acc = torch.mean((predicted_label == label.data).float()) #计算准确率
            acc_.append(acc.data.item())
            
            self.lr = self.optimizer.param_groups[0]['lr'] 
            process.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.data.item(), self.lr)) #输出当前的loss和lr
        
        self.print_log('\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_), np.mean(acc_)*100))
        
        if save_model and epoch%self.args.save_interval==0: #每save_interval个epoch保存模型，默认5
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.args.save_dir + '/epoch_' + str(epoch+1) + '.pt')
    ```
    ```self.model.train()```: 把模型设置为训练模式，对部分层有影响，如```Dropout```或```BN```等。<br>
    主要训练代码：
    ```python
            output = self.model(data) #模型输出
            loss = self.loss(output, label) #计算loss 
            self.optimizer.zero_grad() #梯度初始化为0
            loss.backward() #反向传播得到梯度
            self.optimizer.step() #通过梯度下降法执行参数更新
    ```
    如果想保存模型，通过```model.state_dict()```提取权重，并使用```torch.save(weights, dir)```保存。
    <br>
    
+ evaluate(): 测试模型
    ```python
    def evaluate(self, epoch):
        self.model.eval()
        ...
    ```
    与训练过程类似，只是不做反向传播操作。<br><br>
    
+ start(): 训练或测试
    ```python
    def start(self):
        if self.args.phase == 'train':
            for epoch in range(self.args.start_epoch, self.args.num_epoch):
                self.print_log('*'*100)
                self.train(epoch)
                if epoch % 5 == 0:
                    self.evaluate(epoch)
                self.print_log('Best_Accuracy: {:.2f}%, epoch: {}'.format(self.best_acc, self.best_acc_epoch))
                ...
        
        elif self.args.phase == 'test':
            if self.args.weights is None:
                raise ValueError('Please appoint --weights.')
            self.evaluate(epoch=0)
            ...

    ```
    若```args.phase```设置为```train```，开始训练。其中将每5个epoch测试一次，并保存最佳准确率。<br>
    若```args.phase```设置为```test```，开始测试，另外需要加载```args.weights```。

# **Main**  <br>
---
```python
if __name__ == '__main__':
    parser = get_parser()
    os.chdir(os.getcwd())
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args() #加载config
    init_seed(args.seed) #设置随机种子
    processor = Processor(args) #定义processor
    processor.start() #进行训练或测试
```
从```config.yaml```文件加载超参数，并传入到我们的proccessor，开始训练或测试。  
<br>
