import os
import os.path as osp
import torch
from PIL import Image
import time

from torch.utils.data import Dataset
from torchvision import transforms
from utils.transforms import *

video_root = "/home/haodong/Data/CSL_Isolated/color_video_125000"
csv_root = '/home/liweijie/projects/ActionRecognition/csv/isl'

class Isl(Dataset):

    def __init__(self, setname, mode='train', length=16):
        self.length = length
        csv_path = osp.join(csv_root, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []

        for l in lines:
            name, lb = l.split(',')
            path = osp.join(video_root, name)
            lb = int(lb)
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        scale_size = 256
        crop_size = 224
        input_mean = [0.485, 0.456, 0.406]
        input_std = [0.229, 0.224, 0.225] 
        if mode=='train':
            self.transforms = transforms.Compose([
                GroupScale(int(scale_size)),
                GroupRandomCrop(crop_size),
                To3dFormatTensor(div=True),
                GroupNormalize(input_mean,input_std),
            ])
        else:
            self.transforms = transforms.Compose([
                GroupScale(int(scale_size)),
                GroupCenterCrop(crop_size),
                Stack(roll=False),
                To3dFormatTensor(div=True),
                GroupNormalize(input_mean,input_std),
            ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        start = time.time()
        path, label = self.data[i], self.label[i]
        data = self.read_video(path)
        end = time.time()
        # print('%.3f s'%(end-start))
        return data, label

    def read_video(self, path):
        image_list = os.listdir(path)
        image_list.sort()
        # Ignore the first frame
        image_list = image_list[1:]
        n_frames = len(image_list)
        indices =  self.select_indices(n_frames)
        images = []
        for ind in indices:
            img = Image.open(osp.join(path,image_list[ind])).convert('RGB')
            images.append(img)
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
    dataset = Isl('trainvaltest')
    # Check every file in the dataset
    for i in range(len(dataset)):
        dataset[i]