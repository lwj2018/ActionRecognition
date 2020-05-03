import torch
import torch.nn as nn
import torchvision.models.resnet

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


class P3DBlock_A(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(P3DBlock_A, self).__init__()

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False, stride=stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Spatial kernel
        self.conv2 = nn.Sequential(
            Conv3DSpatial(planes, planes),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Temporal kernel
        self.conv3 = nn.Sequential(
            Conv3DTemporal(planes, planes),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # 1x1x1
        self.conv4 = nn.Sequential(
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
        out = self.conv4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class P3DBlock_B(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(P3DBlock_B, self).__init__()

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False, stride=stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Spatial kernel
        self.conv2 = nn.Sequential(
            Conv3DSpatial(planes, planes),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Temporal kernel
        self.conv3 = nn.Sequential(
            Conv3DTemporal(planes, planes),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # 1x1x1
        self.conv4 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        outs = self.conv2(out)
        outt = self.conv3(out)
        out = outs + outt
        out = self.conv4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class P3DBlock_C(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(P3DBlock_C, self).__init__()

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False, stride=stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Spatial kernel
        self.conv2 = nn.Sequential(
            Conv3DSpatial(planes, planes),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Temporal kernel
        self.conv3 = nn.Sequential(
            Conv3DTemporal(planes, planes),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # 1x1x1
        self.conv4 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        outs = self.conv2(out)
        outst = self.conv3(outs)
        out = outs + outst
        out = self.conv4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """
    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

class VideoResNet(nn.Module):

    def __init__(self, block, layers,
                 stem, num_classes=400,
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
        super(VideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_optim_policies(self,lr):
        return [
            {'params':self.stem.parameters(),'lr':lr},
            {'params':self.layer1.parameters(),'lr':lr},
            {'params':self.layer2.parameters(),'lr':lr},
            {'params':self.layer3.parameters(),'lr':lr},
            {'params':self.layer4.parameters(),'lr':lr},
            {'params':self.fc.parameters(),'lr':lr}
        ]


def _video_resnet(arch, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)

    if pretrained:
        # part resume
        model_dict = model.state_dict()
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        state_dict = {k : v for k,v in state_dict.items() if 'fc' not in k}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    return model

def p3d18_a(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Pseudo-3D model as in
    http://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: P3DA-18 network
    """

    return _video_resnet('p3d_a',
                         pretrained, progress,
                         block=P3DBlock_A,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)

def p3d18_b(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Pseudo-3D model as in
    http://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: P3DB-18 network
    """

    return _video_resnet('p3d_b',
                         pretrained, progress,
                         block=P3DBlock_B,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)

def p3d18_c(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Pseudo-3D model as in
    http://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: P3DC-18 network
    """

    return _video_resnet('p3d_c',
                         pretrained, progress,
                         block=P3DBlock_C,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)
