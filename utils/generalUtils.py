from datasets.isl import Isl
from models.resnet3d import r3d_18
from models.resnet3d import r2plus1d_18
from torch.utils.data import DataLoader

def getNumclass(dataset):
    if dataset == 'isl':
        num_class = 500
    return num_class

def getDataloader(dataset,args):
    if dataset == 'isl':
        trainset = Isl('trainval')
        valset = Isl('test')
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
    return model
