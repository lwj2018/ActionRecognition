import torch
import torch.nn.functional as F
import numpy
import time
from utils.metricUtils import *
from utils.Averager import AverageMeter
from utils.Recorder import Recorder

def eval_c3d(model, criterion, valloader, 
        device, epoch, log_interval, writer, eval_samples):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_top1 = AverageMeter()
    avg_top5 = AverageMeter()
    averagers = [losses, avg_top1, avg_top5]
    names = ['val loss','val top1','val top5']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set evaluation mode
    model.eval()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(valloader):
        with torch.no_grad():
            # Reduce the evaluation time !!!
            if i>eval_samples: break
            # measure data loading time
            recoder.data_tok()

            # get the data and labels
            data,lab = [_.to(device) for _ in batch]

            # forward
            outputs = model(data)

            # compute the loss
            loss = criterion(outputs,lab)

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
        if i==0 or i % log_interval == log_interval-1 or i==len(valloader)-1:
            recoder.log(epoch,i,len(valloader),mode='Eval')
        
    return recoder.get_avg('val top1')

def eval_tsn(model, criterion, valloader, 
        device, epoch, log_interval, writer, eval_samples):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_top1 = AverageMeter()
    avg_top5 = AverageMeter()
    averagers = [losses, avg_top1, avg_top5]
    names = ['val loss','val top1','val top5']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set evaluation mode
    model.eval()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(valloader):
        with torch.no_grad():
            # Reduce the evaluation time !!!
            if i>eval_samples: break
            # measure data loading time
            recoder.data_tok()

            # get the data and labels
            data,flow_data,lab = [_.to(device) for _ in batch]

            # forward
            outputs = model(data,flow_data)

            # compute the loss
            loss = criterion(outputs,lab)

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
        if i==0 or i % log_interval == log_interval-1 or i==len(valloader)-1:
            recoder.log(epoch,i,len(valloader),mode='Eval')
        
    return recoder.get_avg('val top1')