import os, sys
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn,utils 
import mxnet.ndarray as F
from tqdm import trange, tqdm

from models import *
from utils import *
from data_loader import *


# set gpu count
def setting_ctx(GPU_COUNT):
    if GPU_COUNT > 0 :
        ctx = [mx.gpu(i) for i in range(GPU_COUNT)]
    else :
        ctx = [mx.cpu()]
    return ctx

class Train(object):
    def __init__(self, config):
        ##setting hyper-parameters
        self.config = config
        self.batch_size = config.batch_size
        self.epoches = config.epoches
        self.N = config.N
        self.K = config.K
        self.input_dims = config.input_dims
        self.GPU_COUNT = config.GPU_COUNT
        self.ctx = setting_ctx(self.GPU_COUNT)
        self.build_model()
        
    def build_model(self):
        self.net = SNAIL(N=self.N,K=self.K,input_dims=self.input_dims,ctx=self.ctx)
        self.net.collect_params().initialize(ctx=self.ctx)
        self.loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
        self.trainer = gluon.Trainer(self.net.collect_params(),optimizer='Adam',optimizer_params={'learning_rate':0.0001})
        
    def save_model(self,epoch,tr_acc,te_acc):
        filename = 'models/best_perf_epoch_'+str(epoch)+"_tr_acc_"+str(te_acc)+"_te_acc_"+str(te_acc)
        self.net.save_params(filename)
    
    def train(self):
        tr_dataloader, te_dataloader = loader(self.config,self.ctx)
        tr_acc_per_epoch = list()
        te_acc_per_epoch = list()
        train_acc = mx.metric.Accuracy()
        test_acc = mx.metric.Accuracy()
        global_va_acc = 0.0
        for epoch in trange(self.epoches):
            tr_acc = list()
            te_acc = list()
            tr_iter = iter(tr_dataloader)
            te_iter = iter(te_dataloader)
            for batch in tqdm(tr_iter):
                x, y = batch
                x, y, last_targets = batch_for_few_shot(self.N, self.K ,self.batch_size, x, y)       
                with autograd.record():
                    x_split = gluon.utils.split_and_load(x,self.ctx)
                    y_split = gluon.utils.split_and_load(y,self.ctx)
                    last_targets_split = gluon.utils.split_and_load(last_targets,self.ctx)
                    last_model = [self.net(X,Y)[:,-1,:] for X, Y in zip(x_split,y_split)]
                    loss_val = [self.loss_fn(X,Y) for X, Y in zip(last_model,last_targets_split)]
                    for l in loss_val:
                        l.backward()
                    for pred,target in zip(last_model,last_targets_split):
                        train_acc.update(preds=nd.argmax(pred,1),labels=target)
                        tr_acc.append(train_acc.get()[1])
                self.trainer.step(self.batch_size,ignore_stale_grad=True)
        
            for batch in tqdm(te_iter):
                x,y = batch
                x, y, last_targets = batch_for_few_shot(self.N, self.K ,int(self.batch_size / len(self.ctx)), x, y)
                x = x.copyto(self.ctx[0])
                y = y.copyto(self.ctx[0])
                last_targets = last_targets.copyto(self.ctx[0])
                model_output = self.net(x,y)
                last_model = model_output[:,-1,:]
                test_acc.update(preds=nd.argmax(last_model,1),labels=last_targets)
                te_acc.append(test_acc.get()[1])
            current_va_acc = np.mean(te_acc) 
            if global_va_acc < current_va_acc:
                self.save_model(epoch,round(np.mean(tr_acc),2),round(np.mean(te_acc),2))
                global_va_acc = current_va_acc
            print("epoch {e}  train_acc:{ta} test_acc:{tea} ".format(e=epoch,ta=np.mean(tr_acc),tea=np.mean(te_acc)))
            tr_acc_per_epoch.append(np.mean(tr_acc))
            te_acc_per_epoch.append(np.mean(te_acc))

        