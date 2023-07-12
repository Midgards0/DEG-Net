import os, time
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import network, loss
import argparse
from torch.autograd import Variable
import os.path as osp
import HSIC
from data_list import ImageList,GroupData,Group24Sampler
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])
def data_load(opt): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = opt['batch_size']
    txt_src = open(opt['train_dset_path']).readlines()
    txt_test = open(opt['test_dset_path']).readlines()


    dsize = len(txt_src)
    tr_size = int(0.9*dsize)
    _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    tr_txt = txt_src

    dsets["train"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["train"] = DataLoader(dsets["train"], batch_size=opt['n_target_samples'], shuffle=False, num_workers=opt['worker'], drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*4, shuffle=True, num_workers=opt['worker'], drop_last=False)

    return dset_loaders


def beta(epoch):
    return 2/(1+np.exp(-10*epoch))-1

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def train_traget(opt):
    #load data
    dset_loaders = data_load(opt)
    #load model
    netF = network.ResBase(res_name=opt['net']).cuda()
    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=opt['bottleneck']).cuda()
    netC = network.feat_classifier(type='wn', class_num = opt['classes_num'], bottleneck_dim=opt['bottleneck']).cuda()
    
    netF.load_state_dict(torch.load(osp.join(opt['output'], "source_F.pt")))
    netB.load_state_dict(torch.load(osp.join(opt['output'], "source_B.pt")))
    netC.load_state_dict(torch.load(osp.join(opt['output'], "source_C.pt")))

#------------------generate fake data & FSDA--------------------------
    G = network.Generator(opt['generator_dim'],input=opt['g_input_dim'])   
    G.weight_init(mean=0.0, std=0.02)
    G = G.cuda()
    print(opt['g_input_dim'])
    G_optimizer=optim.Adam(G.parameters(), lr=opt['lr']*0.1, betas=(0.5, 0.999))
    G_losses = 0
    save_acc =[]

    def metric(G_result, X, n, batch_size):
        all_sum =0
        for i in range(batch_size):
            sum_abs_diff = 0
            for j in range(n):
                abs_diff = torch.abs(G_result[i].view(-1) - X[j].view(-1))
                w = abs_diff / torch.norm(abs_diff, p=2, keepdim=False)
                sum_abs_diff += torch.dot(w, abs_diff)
            sum_abs_diff = sum_abs_diff/n
            all_sum += sum_abs_diff
        all_sum = all_sum/batch_size
        return all_sum

    
    for epoch in range(1, opt['generate_epoch']+1):
        netF = netF.eval()
        netB = netB.eval()
        netC = netC.eval()
        G=G.train()
        X_s = torch.Tensor(opt['classes_num']*opt['generate_batch'],opt['data_channel'],opt['data_size'],opt['data_size'])
        Y_s = torch.LongTensor(opt['classes_num']*opt['generate_batch'])
        G_loss=[0]*3
        G_optimizer.zero_grad()
        iter_train = iter(dset_loaders["train"])
       
        for target in range(opt['classes_num']):
            X_tar, Y_tar = iter_train.next()
            label_ = np.ones(opt['generate_batch'],dtype=np.int32)*target
            class_onehot = np.zeros((opt['generate_batch'],opt['classes_num']))
            class_onehot[np.arange(opt['generate_batch']), label_] = 1
            z_ = np.random.normal(0, 1, (opt['generate_batch'], opt['g_input_dim']))
            z_[np.arange(opt['generate_batch']), :opt['classes_num']] = class_onehot[np.arange(opt['generate_batch'])]
            z_ = (torch.from_numpy(z_).float())
            z_=z_.view(opt['generate_batch'],  opt['g_input_dim'], 1, 1)
            z_ = Variable(z_.cuda())
       
            G_result = G(z_)
            G_result = G_result.cuda()
           
            s_g = netB(netF(G_result))
            logits = netC(s_g)
            
          
            #calculate loss
            X_tar=X_tar.cuda()
            s_t=netB(netF(X_tar))
            sim_loss = metric(s_g, s_t, opt['n_target_samples'], opt['generate_batch'])
            
            ones = torch.ones(opt['generate_batch'])
            ones = ones.cuda()
            con_loss = MSELoss(logits[:,target], ones)
            
            div_loss = torch.sqrt(HSIC.hsic_normalized_cca(s_g,s_g,sigma=opt['sigma']))

            #print(sim_loss,con_loss,div_loss)
            
            G_loss[0]=G_loss[0]+con_loss
            G_loss[1]=G_loss[1]+sim_loss
            G_loss[2]=G_loss[2]+div_loss
           
            
        

        G_losses=G_loss[0]+opt['lambda']*G_loss[1]+opt['beta']*G_loss[2]
        G_losses=G_losses/opt['classes_num']
        G_losses.backward()
        G_optimizer.step()
        if epoch%50 == 0:
            print('[%d/%d]    loss_g: %.3f    con_loss:%.3f    sim_loss:%.3f  div_loss:%.3f' % (epoch, opt['generate_epoch'], G_losses,G_loss[0]/opt['classes_num'],G_loss[1]/opt['classes_num'],G_loss[2]/opt['classes_num']))
        #generate data
        G=G.eval()
        txt_src = open(opt['train_dset_path']).readlines()
        group_data=GroupData(txt_src,G,opt['generate_batch'],opt['classes_num'],opt['n_target_samples'],transform=image_train())
        
        if epoch == opt['generate_epoch']-opt['model_epoch']:
            #random
           
            discriminator = network.DCD(input_features=opt['gd_dim'])
            discriminator = discriminator.cuda()
            discriminator.train()
            loss_fn = torch.nn.CrossEntropyLoss()

            optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=opt['lr']*0.3)
        
            
            for epoch_2 in range(opt['gd_epoch']):
                
                group_data.shuffle(seed=epoch_2)
                group_loader=DataLoader(group_data, batch_size=40, shuffle=True, num_workers=opt['worker'], drop_last=True)
                # data
                loss_mean=[]
                for data1,data2,label1,label2,group_label in group_loader:
                    data1=data1.cuda()
                    data2=data2.cuda()
                    group_label=group_label.cuda()

                    optimizer_D.zero_grad()
                    X_cat=torch.cat([netB(netF(data1)),netB(netF(data2))],1)
                    y_pred=discriminator(X_cat.detach())
                    loss=loss_fn(y_pred,group_label)
                    loss.backward()
                    optimizer_D.step()
                    loss_mean.append(loss.item())

                print("pretrain group discriminator----Epoch %d/%d loss:%.3f"%(epoch_2+1,opt['gd_epoch'],np.mean(loss_mean)))
                
        g_h_loss_mean=[]
        d_loss_mean=[]
        if epoch > opt['generate_epoch']-opt['model_epoch']:
            netF.train()
            netB.train()
            netC.train()
            discriminator.eval()
            
            optimizer_g_h=torch.optim.Adam(list(netF.parameters())+list(netB.parameters())+list(netC.parameters()),lr=opt['lr']*4)
            optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=opt['lr']*0.3)

            scheduler_g_h=torch.optim.lr_scheduler.StepLR(optimizer_g_h,step_size=20,gamma=0.1)
            scheduler_d=torch.optim.lr_scheduler.StepLR(optimizer_d,step_size=20,gamma=0.1)

            #---training g and h , group discriminator is frozen
            group_data.shuffle(seed=opt['gd_epoch']+epoch)
            g24_sampler=Group24Sampler(opt['classes_num'],opt['n_target_samples'])
            group24_loader=DataLoader(group_data, batch_size=40, num_workers=opt['worker'],sampler=g24_sampler)
            for data1,data2,label1,label2,group_label in group24_loader:
                data1=data1.cuda()
                data2=data2.cuda()
                label1=label1.cuda()
                label2=label2.cuda()
                group_label=group_label.cuda()

                optimizer_g_h.zero_grad()
                s_1=netB(netF(data1))
                s_2=netB(netF(data2))
                s_cat=torch.cat([s_1,s_2],1)

                y_pred1=netC(s_1)
                y_pred2=netC(s_2)
                y_pred_dcd=discriminator(s_cat)
                loss_t=loss_fn(y_pred2,label2)
                loss_dcd=loss_fn(y_pred_dcd,group_label)
                loss_sum =  loss_t + beta(epoch-opt['generate_epoch']+opt['gd_epoch']) * loss_dcd #
                loss_sum.backward()
                g_h_loss_mean.append(loss_sum.item())  
                optimizer_g_h.step()
                scheduler_g_h.step()
            #----training group discriminator ,g and h frozen

            netF.eval()
            netB.eval()
            netC.eval()
            discriminator.train()
            group_loader=DataLoader(group_data, batch_size=40, shuffle=True, num_workers=opt['worker'], drop_last=True)
            
            dcd_loss_mean=[]
            for data1,data2,label1,label2,group_label in group_loader:
                data1=data1.cuda()
                data2=data2.cuda()
                group_label=group_label.cuda()

                optimizer_D.zero_grad()
                X_cat=torch.cat([netB(netF(data1)),netB(netF(data2))],1)
                y_pred=discriminator(X_cat.detach())
                loss=loss_fn(y_pred,group_label)
                loss.backward()
                optimizer_D.step()
                dcd_loss_mean.append(loss.item())

            print("step3----Epoch %d/%d    g_h_loss: %.3f    d_loss: %.3f " % (epoch, opt['generate_epoch'],np.mean(g_h_loss_mean),np.mean(dcd_loss_mean)))
            #del generated data
            torch.cuda.empty_cache()


            if epoch%10  ==0:
                acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = '\n Accuracy = {:.2f}%'.format(acc) + '\n' + acc_list
                
                print(log_str)

            

           

           
        
parser=argparse.ArgumentParser()

parser.add_argument('--gd_epoch',type=int,default=100)
parser.add_argument('--model_epoch',type=int,default=100)
parser.add_argument('--generate_epoch',type=int,default=500)
parser.add_argument('--n_target_samples',type=int,default=6)
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--generate_batch',type=int,default=13)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--seed',type=int,default=1)
parser.add_argument('--generator_dim',type=int,default=64)
parser.add_argument('--gz_dim',type=int,default=100)
parser.add_argument('--worker', type=int, default=4, help="number of workers")
parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
parser.add_argument('--bottleneck', type=int, default=256)
parser.add_argument('--gd_dim',type=int,default=256*2)
parser.add_argument('--data_size',type=int,default=224)
parser.add_argument('--data_channel',type=int,default=3)
parser.add_argument('--lambda',type=float,default=0.2)
parser.add_argument('--beta',type=float,default=0.07)
parser.add_argument('--sigma',type=float,default=5.)
parser.add_argument('--output', type=str, default='result')
opt=vars(parser.parse_args())

use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')

torch.manual_seed(opt['seed'])
if use_cuda:
    torch.cuda.manual_seed(opt['seed'])

l1loss = nn.L1Loss().cuda()
MSELoss = nn.MSELoss().cuda()
opt['classes_num']=12
opt['g_input_dim']=opt['classes_num']+opt['gz_dim']
#dir
#opt['name_src'] = names[args.s][0].upper()
if not osp.exists(opt['output']):
    os.system('mkdir -p ' + opt['output'])
if not osp.exists(opt['output']):
    os.mkdir(opt['output'])
opt['train_dset_path']='./data/VISDA-C/validation_list_'+str(opt['n_target_samples'])+'_.txt'
opt['test_dset_path']='./data/VISDA-C/validation_list.txt'
train_traget(opt)
