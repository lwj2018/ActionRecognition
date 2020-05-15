# Part of this code refers to https://github.com/zhoubolei/TRN-pytorch
import torch
import torch.nn as nn
import torchvision.models

class tsn(nn.Module):

    def __init__(self,num_classes=500,length=16,
            base_model='resnet18',
            weight=0.5,
            flow_channel=10):
        super(tsn,self).__init__()
        self.num_classes = num_classes
        self.length = length
        self.weight = weight
        self._get_base_model()
        self._adjust_first_layer(flow_channel)

    def forward(self,x,flow):
        x = self.rgbmodel(x)
        flow = self.flowmodel(flow)
        out = x + weight * flow
        return out

    def _get_base_model(self,base_model):
        n = self.num_classes
        if 'resnet' in base_model:
            self.rgbmodel = getattr(torchvision.models,base_model)(pretrained=True,num_classes=n)
            self.flowmodel = getattr(torchvision.models,base_model)(pretrained=True,num_classes=n)

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

