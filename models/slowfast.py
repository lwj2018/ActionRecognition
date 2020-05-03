import torch
import torch.nn as nn

class Conv3DSpatial(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 stride=1,
                 padding=1):

        super(Conv3DSpatial, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)


class Conv3DTemporal(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 stride=1,
                 padding=1):

        super(Conv3DTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 1, 1),
            stride=(stride, 1, 1),
            padding=(padding, 0, 0),
            bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):

        super(Bottleneck, self).__init__()

        # 1x1x1
        if head_conv == 1:
            self.conv1 = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
                nn.BatchNorm3d(planes),
                nn.ReLU(inplace=True)
            )
        elif head_conv == 3:
            self.conv1 = nn.Sequential(
                Conv3DTemporal(inplanes, planes),
                nn.BatchNorm3d(planes),
                nn.ReLU(inplace=True)
            )
        # Spatial kernel
        self.conv2 = nn.Sequential(
            Conv3DSpatial(planes, planes, stride=stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FastStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(FastStem, self).__init__(
            nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2),
                      padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))

class SlowStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(SlowStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                      padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))

class SlowFast(nn.Module):

    def __init__(self, block, layers,
                 num_classes=400,
                 dropout=0.5,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(SlowFast, self).__init__()

        self.fast_inplanes = 8
        self.fast_stem = FastStem()
        self.fast_layer1 = self._make_layer_fast(block, 8, layers[0], stride=1, head_conv=3)
        self.fast_layer2 = self._make_layer_fast(block, 16, layers[1], stride=2, head_conv=3)
        self.fast_layer3 = self._make_layer_fast(block, 32, layers[2], stride=2, head_conv=3)
        self.fast_layer4 = self._make_layer_fast(block, 64, layers[3], stride=2, head_conv=3)

        self.lateral_p1 = nn.Conv3d(8, 8*2, kernel_size=(5, 1, 1), stride=(8, 1 ,1), bias=False, padding=(2, 0, 0))
        self.lateral_res2 = nn.Conv3d(32,32*2, kernel_size=(5, 1, 1), stride=(8, 1 ,1), bias=False, padding=(2, 0, 0))
        self.lateral_res3 = nn.Conv3d(64,64*2, kernel_size=(5, 1, 1), stride=(8, 1 ,1), bias=False, padding=(2, 0, 0))
        self.lateral_res4 = nn.Conv3d(128,128*2, kernel_size=(5, 1, 1), stride=(8, 1 ,1), bias=False, padding=(2, 0, 0))

        self.slow_inplanes = 64+64//8*2
        self.slow_stem = SlowStem()
        self.slow_layer1 = self._make_layer_slow(block, 64, layers[0], stride=1, head_conv=1)
        self.slow_layer2 = self._make_layer_slow(block, 128, layers[1], stride=2, head_conv=1)
        self.slow_layer3 = self._make_layer_slow(block, 256, layers[2], stride=2, head_conv=3)
        self.slow_layer4 = self._make_layer_slow(block, 512, layers[3], stride=2, head_conv=3)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(self.fast_inplanes+512 * block.expansion, num_classes, bias=False)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        fast, lateral = self.fastPath(x[:, :, :, :, :])
        slow = self.slowPath(x[:, :, ::8, :, :], lateral)

        x = torch.cat([slow, fast], dim=1)
        x = self.dp(x)
        x = self.fc(x)
        return x

    def fastPath(self, x):
        lateral = []
        x = self.fast_stem(x)
        lateral_p = self.lateral_p1(x)
        lateral.append(lateral_p)

        x = self.fast_layer1(x)
        lateral_res2 = self.lateral_res2(x)
        lateral.append(lateral_res2)

        x = self.fast_layer2(x)
        lateral_res3 = self.lateral_res3(x)
        lateral.append(lateral_res3)

        x = self.fast_layer3(x)
        lateral_res4 = self.lateral_res4(x)
        lateral.append(lateral_res4)

        x = self.fast_layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)

        return x,lateral

    def slowPath(self, x, lateral):
        x = self.slow_stem(x)
        x = torch.cat([x,lateral[0]],dim=1)
        x = self.slow_layer1(x)
        x = torch.cat([x,lateral[1]],dim=1)
        x = self.slow_layer2(x)
        x = torch.cat([x,lateral[2]],dim=1)
        x = self.slow_layer3(x)
        x = torch.cat([x,lateral[3]],dim=1)
        x = self.slow_layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None

        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.fast_inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1,stride,stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.fast_inplanes, planes, stride, downsample, head_conv=head_conv))

        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes, head_conv=head_conv))

        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None

        if stride != 1 or self.slow_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.slow_inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1,stride,stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.slow_inplanes, planes, stride, downsample, head_conv=head_conv))

        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes, head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion + planes * block.expansion//8*2

        return nn.Sequential(*layers)

    def get_optim_policies(self,lr):
        return [
            {'params':self.fast_stem.parameters(),'lr':lr},
            {'params':self.fast_layer1.parameters(),'lr':lr},
            {'params':self.fast_layer2.parameters(),'lr':lr},
            {'params':self.fast_layer3.parameters(),'lr':lr},
            {'params':self.fast_layer4.parameters(),'lr':lr},
            {'params':self.lateral_p1.parameters(),'lr':lr},
            {'params':self.lateral_res2.parameters(),'lr':lr},
            {'params':self.lateral_res3.parameters(),'lr':lr},
            {'params':self.lateral_res4.parameters(),'lr':lr},
            {'params':self.slow_stem.parameters(),'lr':lr},
            {'params':self.slow_layer1.parameters(),'lr':lr},
            {'params':self.slow_layer2.parameters(),'lr':lr},
            {'params':self.slow_layer3.parameters(),'lr':lr},
            {'params':self.slow_layer4.parameters(),'lr':lr},
            {'params':self.fc.parameters(),'lr':lr}
        ]

def _video_slowfast(arch, pretrained=False, progress=True, **kwargs):
    model = SlowFast(**kwargs)

    if pretrained:
        # part resume
        model_dict = model.state_dict()
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        state_dict = {k : v for k,v in state_dict.items() if 'fc' not in k}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    return model

def slowfast18(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer SlowFast model as in
    http://openaccess.thecvf.com/content_ICCV_2019/papers/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.pdf

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: SlowFast-18 network
    """

    return _video_slowfast('slowfast18',
                         pretrained, progress,
                         block=Bottleneck,
                         layers=[2, 2, 2, 2],
                         **kwargs)

def slowfast18(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer SlowFast model as in
    http://openaccess.thecvf.com/content_ICCV_2019/papers/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.pdf

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: SlowFast-18 network
    """

    return _video_slowfast('slowfast18',
                         pretrained, progress,
                         block=Bottleneck,
                         layers=[2, 2, 2, 2],
                         **kwargs)

def slowfast50(pretrained=False, progress=True, **kwargs):
    """Construct 50 layer SlowFast model as in
    http://openaccess.thecvf.com/content_ICCV_2019/papers/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.pdf

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: SlowFast-50 network
    """

    return _video_slowfast('slowfast18',
                         pretrained, progress,
                         block=Bottleneck,
                         layers=[3, 4, 6, 3],
                         **kwargs)

def slowfast101(pretrained=False, progress=True, **kwargs):
    """Construct 101 layer SlowFast model as in
    http://openaccess.thecvf.com/content_ICCV_2019/papers/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.pdf

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: SlowFast-101 network
    """

    return _video_slowfast('slowfast18',
                         pretrained, progress,
                         block=Bottleneck,
                         layers=[3, 4, 23, 3],
                         **kwargs)

def slowfast152(pretrained=False, progress=True, **kwargs):
    """Construct 152 layer SlowFast model as in
    http://openaccess.thecvf.com/content_ICCV_2019/papers/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.pdf

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: SlowFast-152 network
    """

    return _video_slowfast('slowfast18',
                         pretrained, progress,
                         block=Bottleneck,
                         layers=[3, 8, 36, 3],
                         **kwargs)

