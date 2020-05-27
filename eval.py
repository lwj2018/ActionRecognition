import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.generalUtils import *
from utils.ioUtils import *
from tensorboardX import SummaryWriter
from utils.trainUtils import train_c3d
from utils.testUtils import eval_c3d
from thop import profile,clever_format
class Arguments:
    def __init__(self):
        # Hyper params
        self.epochs = 10
        self.learning_rate = 1e-5 # default 1e-5
        self.batch_size = 2
        # Train settings
        self.num_workers = 1
# Options
dataset = 'isl'
num_class = getNumclass(dataset)
model_type = 'r2plus1d'
store_name = '_'.join(['eval',dataset,model_type])
summary_name = '/data/projects/ActionRecognition/runs/' + store_name
checkpoint = '/data/projects/ActionRecognition/checkpoint/isl_r2plus1d_best.pth.tar'
log_interval = 100
device_list = '1,2'
model_path = "/data/projects/ActionRecognition/checkpoint"
create_path(model_path)

start_epoch = 0
best_acc = 0.00
# get args
args = Arguments()
# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare dataset & dataloader
# Shape of input: N x 3 x 16 x 224 x 224
train_loader, val_loader = getDataloader(dataset,args)

# Build model
model = getModel(model_type,num_class).to(device)
# Run the model parallelly
if checkpoint is not None:
    start_epoch, best_acc = resume_model(model, checkpoint)
if torch.cuda.device_count() > 1:
    print("------- Using {} GPUs --------".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)

# Create loss criterion
criterion = nn.CrossEntropyLoss()

# Use writer to record
writer = SummaryWriter(os.path.join(summary_name, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

# Start Evaluation
print("Eval %s on %s"%(model_type,dataset))
print("Evaluation Started".center(60, '#'))
for epoch in range(start_epoch, start_epoch + 1):
    # Eval the model
    acc = eval_c3d(model,criterion,val_loader,device,epoch,log_interval,writer,eval_samples=len(val_loader))
    print('Epoch best acc: {:.3f}'.format(acc))

print("Evaluation Finished".center(60, '#'))