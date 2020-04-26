import cv2
import os
from glob import glob
import numpy as np
import time

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def cal_for_frames(video_path):
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()
 
    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames[1:]):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr
 
    return flow
 
 
def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow.
    Limit the range of flow to (-bound, bound)
    """
    TVL1=cv2.optflow.DualTVL1OpticalFlow_create()
    start=time.time()
    flow = TVL1.calc(prev, curr, None)
    end=time.time()
    print('cal one frame cost: %.3f s'%(end-start))
    assert flow.dtype == np.float32
 
    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
 
    return flow
 
 
def save_flow(video_flows, flow_path):
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path,"x_{:06d}.jpg".format(i)),
                    flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path,"y_{:06d}.jpg".format(i)),
                    flow[:, :, 1])
 
 
def extract_flow(video_path, flow_path):
    start = time.time()
    flow = cal_for_frames(video_path)
    end = time.time()
    print('Calc flow cost: %.3f s'%(end-start))
    # Calc flow of a video about 100 frames cost about 50 s
    save_flow(flow, flow_path)
    end = time.time()
    print('Save cost: %.3f s'%(end-start))
    # Save cost about 5 s
    print('complete:' + flow_path)
    return
 
 
if __name__ == '__main__':
    video_root='/home/haodong/Data/CSL_Isolated/color_video_125000'
    flow_root='/home/liweijie/NFS/Data/CSL_Isolated/flow_125000'
    folder_list = glob(video_root+'/*')
    folder_list.sort()
    for i,folder_path in enumerate(folder_list):
        folder = folder_path.split('/')[-1]
        save_folder_path = os.path.join(flow_root,folder)
        create_path(save_folder_path)
        video_list = glob(folder_path+'/*')
        video_list.sort()
        for video_path in video_list:
            save_path = os.path.join(flow_root,folder,video_path.split('/')[-1])
            create_path(save_path)
            extract_flow(video_path,save_path)
