import os
import os.path as osp
import numpy

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

num_class = 500
n_train_person = 36
csv_path = '../csv/isl/'
create_path(csv_path)
video_root = "/home/haodong/Data/CSL_Isolated/color_video_125000"
trainvaltest_csv = open(csv_path+'trainvaltest.csv','w')
trainvaltest_csv.write('foldername,label\n')
trainval_csv = open(csv_path+'trainval.csv','w')
trainval_csv.write('foldername,label\n')
test_csv = open(csv_path+'test.csv','w')
test_csv.write('foldername,label\n')
for c in range(num_class):
    print('%d/%d'%(c,num_class))
    c_folder = '%03d'%c
    c_path = osp.join(video_root,c_folder)
    video_list = os.listdir(c_path)
    video_list.sort()
    for video in video_list:
        person = int(video.split('_')[0].lstrip('P'))
        video_path = osp.join(c_folder,video)
        record = video_path + ',' + str(c) + '\n'
        trainvaltest_csv.write(record)
        if person <= n_train_person:
            trainval_csv.write(record)
        else:
            test_csv.write(record)

