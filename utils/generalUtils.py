from datasets import Isl,Isl_RGBflow
from models.resnet3d import r3d_18,r2plus1d_18
from models.p3d import p3d18_a,p3d18_b,p3d18_c
from models.slowfast import slowfast18,slowfast50,slowfast101,slowfast152
from models.tsn import tsn
from torch.utils.data import DataLoader

def getNumclass(dataset):
    if dataset == 'isl':
        num_class = 500
    else:
        raise Exception('Dont support this dataset: %s'%dataset)
    return num_class

def getDataloader(dataset,args):
    if dataset == 'isl':
        trainset = Isl('trainval')
        valset = Isl('test')
    if dataset == 'isl_rgbflow':
        trainset = Isl_RGBflow('trainval')
        valset = Isl_RGBflow('test')
    else:
        raise Exception('Dont support this dataset: %s'%dataset)
    train_loader = DataLoader(trainset,batch_size=args.batch_size,shuffle=True,
        num_workers=args.num_workers,pin_memory=True)
    val_loader = DataLoader(trainset,batch_size=args.batch_size,shuffle=True,
        num_workers=args.num_workers,pin_memory=True)
    return train_loader, val_loader

def getModel(model_type,num_class):
    if model_type == 'r3d':
        model = r3d_18(pretrained=True,num_classes=num_class)
    elif model_type == 'r2plus1d':
        model = r2plus1d_18(pretrained=True,num_classes=num_class)
    elif model_type == 'p3d18a':
        model = p3d18_a(pretrained=False,num_classes=num_class)
    elif model_type == 'p3d18b':
        model = p3d18_b(pretrained=False,num_classes=num_class)
    elif model_type == 'p3d18c':
        model = p3d18_c(pretrained=False,num_classes=num_class)
    elif model_type == 'slowfast18':
        model = slowfast18(pretrained=False,num_classes=num_class)
    elif model_type == 'slowfast50':
        model = slowfast50(pretrained=False,num_classes=num_class)
    elif model_type == 'slowfast101':
        model = slowfast101(pretrained=False,num_classes=num_class)
    elif model_type == 'slowfast152':
        model = slowfast152(pretrained=False,num_classes=num_class)
    elif model_type == 'tsn':
        model = tsn(num_classes=num_class)
    else:
        raise Exception('Dont support this type of model: %s'%model_type)
    return model
