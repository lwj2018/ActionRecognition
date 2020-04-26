import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.generalUtils import *
from utils.ioUtils import *
from torch.utils.tensorboard import SummaryWriter
from utils.trainUtils import train_c3d
from utils.testUtils import eval_c3d

class Arguments:
    def __init__(self):
        # Hyper params
        self.epochs = 10
        self.learning_rate = 1e-5
        self.batch_size = 4
        # Train settings
        self.num_workers = 8
# Options
dataset = 'isl'
num_class = getNumclass(dataset)
model_type = 'r3d'
store_name = '_'.join([dataset,model_type])
summary_name = 'runs/' + store_name
checkpoint = None
log_interval = 100
device_list = '3'
model_path = "./checkpoint"

start_epoch = 0
best_acc = 0.00
# get args
args = Arguments()
# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join(summary_name, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

# Prepare dataset & dataloader
train_loader, val_loader = getDataloader(dataset,args)

# Build model
model = getModel(model_type,num_class).to(device)
if checkpoint is not None:
    start_epoch, best_acc = resume_model(model, checkpoint)

# Create loss criterion & optimizer
criterion = nn.CrossEntropyLoss()
polices = model.get_optim_policies(args.learning_rate)
optimizer = torch.optim.Adam(polices)

# Start training
best_acc = 0.0
print("Training Started".center(60, '#'))
for epoch in range(start_epoch, start_epoch + args.epochs):
    # Train the model
    train_c3d(model,criterion,optimizer,train_loader,device,epoch,log_interval,writer)
    # Eval the model
    acc = eval_c3d(model,criterion,val_loader,device,epoch,log_interval,writer)
    # Save model
    # remember best acc and save checkpoint
    is_best = acc>best_acc
    best_acc = max(acc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best': best_acc,
    }, is_best, model_path, store_name)
    print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))
    print('Epoch best acc: {:.3f}'.format(best_acc))

print("Training Finished".center(60, '#'))