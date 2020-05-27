# Part of this code refers to https://github.com/zhoubolei/TRN-pytorch
import torch
import torch.nn as nn
import models.resnet2d

class tsn(nn.Module):

    def __init__(self,num_classes=500,length=16,
            base_model='resnet18',
            weight=0.5,
            flow_channel=10):
        super(tsn,self).__init__()
        self.num_classes = num_classes
        self.length = length
        self.weight = weight
        self.flow_channel = flow_channel
        self._get_base_model(base_model)
        self._adjust_first_layer(flow_channel)
        self.consensusModel = AvgConsensusModel()

    def forward(self,x,flow):
        # Rgb forward & flow forward
        N = x.size(0)
        x = x.view( (-1,3) + x.size()[-2:] )
        x = self.rgbmodel(x)
        flow = flow.view( (-1,self.flow_channel) + flow.size()[-2:] )
        flow = self.flowmodel(flow)
        # View
        x = x.view(N,-1,x.size()[-1])
        flow = flow.view(N,-1,flow.size()[-1])
        # Fuse & Consensus
        out = x + self.weight * flow
        out = self.consensusModel(out)
        return out

    def _get_base_model(self,base_model):
        n = self.num_classes
        if 'resnet' in base_model:
            self.rgbmodel = getattr(models.resnet2d,base_model)(pretrained=True,num_classes=n)
            self.flowmodel = getattr(models.resnet2d,base_model)(pretrained=True,num_classes=n)

    def _adjust_first_layer(self,input_channel):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.flowmodel.modules())
        first_conv_idx = list(filter( lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))) ))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (input_channel, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(input_channel, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

    def get_optim_policies(self,lr):
        return [
            {'params':self.rgbmodel.parameters(),'lr':lr},
            {'params':self.flowmodel.parameters(),'lr':lr},
        ]

class AvgConsensusModel(nn.Module):
    def __init__(self):
        super(AvgConsensusModel,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.avgpool(x)
        x = x.squeeze(-1)
        return x
