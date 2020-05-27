import os
import os.path as osp
import torch
from PIL import Image
import time

from torch.utils.data import Dataset
from torchvision import transforms
from utils.transforms import *

# # Settings for server251
# video_root = "/home/haodong/Data/CSL_Isolated/color_video_125000"
# flow_root = "/home/liweijie/NFS/Data/CSL_Isolated/flow"
# csv_root = '/home/liweijie/projects/ActionRecognition/csv/isl_rgbflow'

video_root = "/data/Data/CSL_Isolated/color_video_125000"
flow_root = "/data/Data/CSL_Isolated/flow"
csv_root = '/data/projects/ActionRecognition/csv/isl_rgbflow'

class Isl_RGBflow(Dataset):

    def __init__(self, setname, length=16,
            flow_length=5):
        self.length = length
        self.flow_length = flow_length
        csv_path = osp.join(csv_root, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        flow_data = []
        label = []

        for l in lines:
            name, lb = l.split(',')
            path = osp.join(video_root, name)
            flow_path = osp.join(flow_root, name)
            lb = int(lb)
            data.append(path)
            flow_data.append(flow_path)
            label.append(lb)

        self.data = data
        self.flow_data = flow_data
        self.label = label

        scale_size = 256
        crop_size = 224
        input_mean = [0.485, 0.456, 0.406]
        input_std = [0.229, 0.224, 0.225] 
        if setname=='trainval':
            self.transforms = transforms.Compose([
                GroupScale(int(scale_size)),
                GroupRandomCrop(crop_size),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                GroupNormalize(input_mean,input_std),
            ])
        else:
            self.transforms = transforms.Compose([
                GroupScale(int(scale_size)),
                GroupCenterCrop(crop_size),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                GroupNormalize(input_mean,input_std),
            ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        start = time.time()
        path, flow_path, label = self.data[i], self.flow_data[i], self.label[i]
        data, indices = self.read_video(path)
        flow_data = self.read_flow_video(flow_path,indices)
        end = time.time()
        # print('%.3f s'%(end-start))
        return data, flow_data, label

    def read_video(self, path):
        image_list = os.listdir(path)
        image_list.sort()
        # Ignore the first frame
        n_frames = len(image_list)-1
        indices =  self.select_indices(n_frames)
        images = []
        for ind in indices:
            img = Image.open(osp.join(path,image_list[ind])).convert('RGB')
            images.append(img)
        data = self.transforms(images)
        return data, indices

    def read_flow_video(self, path, indices):
        image_list = os.listdir(path)
        image_list.sort()
        l = len(image_list)
        x_list = image_list[:l//2]
        y_list = image_list[l//2:]
        images = []
        for ind in indices:
            p = 0
            for _ in range(self.flow_length):
                if (ind+p+1)<(l//2): p = p+1
                x_img = Image.open(osp.join(path,x_list[ind+p])).convert('L')
                y_img = Image.open(osp.join(path,y_list[ind+p])).convert('L')
                images.extend([x_img,y_img])
        data = self.transforms(images)
        return data

    def select_indices(self, n_frames):
        indices = np.linspace(0,n_frames-1,self.length).astype(int)
        interval = (n_frames-1)//self.length
        if interval>0:
            jitter = np.random.randint(0,interval,self.length)
        else:
            jitter = 0
        jitter = (np.random.rand(self.length)*interval).astype(int)
        indices = np.sort(indices+jitter)
        indices = np.clip(indices,0,n_frames-1)
        return indices

# Test
if __name__ == '__main__':
    dataset = Isl_RGBflow('trainvaltest')
    # Check every file in the dataset
    for i in range(len(dataset)):
        dataset[i]