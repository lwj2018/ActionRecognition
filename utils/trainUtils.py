import torch
import torch.nn.functional as F
import numpy
import time
from utils.metricUtils import *
from utils.Averager import AverageMeter
from utils.Recorder import Recorder

def train_c3d(model, criterion, optimizer, trainloader, 
        device, epoch, log_interval, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_top1 = AverageMeter()
    avg_top5 = AverageMeter()
    # Create recorder
    averagers = [losses, avg_top1, avg_top5]
    names = ['train loss','train top1','train top5']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set trainning mode
    model.train()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(trainloader):
        # measure data loading time
        recoder.data_tok()

        # get the data and labels
        data,lab = [_.to(device) for _ in batch]

        optimizer.zero_grad()
        # forward
        outputs = model(data)

        # compute the loss
        loss = criterion(outputs,lab)

        # backward & optimize
        loss.backward()
        optimizer.step()

        # compute the metrics
        top1, top5 = accuracy(outputs, lab, topk=(1,5))

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [loss.item(),top1,top5]
        recoder.update(vals)

        # logging
        if i==0 or i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(trainloader))
            # Reset average meters 
            recoder.reset()        
