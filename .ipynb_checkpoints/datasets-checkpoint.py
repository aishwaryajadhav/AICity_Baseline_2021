import json
import os
import random
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
from utils import get_logger


def default_loader(path):
    return Image.open(path).convert('RGB')


class CityFlowNLDataset(Dataset):
    def __init__(self, data_cfg,json_path,transform = None,Random= True):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.data_cfg = data_cfg.clone()
        self.crop_area = data_cfg.CROP_AREA
        self.random = Random
        with open(json_path) as f:
            tracks = json.load(f)
        
        self.uuid_to_index = {}
        self.index_to_uuid = {}
        self.list_of_tracks = []

        for i, (k,v) in enumerate(tracks.items()):
            self.uuid_to_index[k] = i
            self.index_to_uuid[i] = k
            self.list_of_tracks.append(v)

        self.targets_ohe, self.target_ind = self.process_track_targets(len(self.list_of_tracks))
        
        self.transform = transform
        self.bk_dic = {}
        self._logger = get_logger()
        
        self.all_indexs = list(self.index_to_uuid.keys())
        self.flip_tag = [False]*len(self.list_of_tracks)
        flip_aug = False
        # print(len(self.all_indexs))
        # if flip_aug:
        #     for i in range(len(self.list_of_tracks)):
        #         text = self.list_of_tracks[i]["nl"]
        #         for j in range(len(text)):
        #             nl = text[j]
        #             if "turn" in nl:
        #                 if "left" in nl:
        #                     self.all_indexs.append(i)
        #                     self.flip_tag.append(True)
        #                     break
        #                 elif "right" in nl:
        #                     self.all_indexs.append(i)
        #                     self.flip_tag.append(True)
        #                     break
        # print(len(self.all_indexs))
        # print("data load")

    def process_track_targets(self, n):
        target_lst = []
        target_ind = []
        max_len = -1
        for track in self.list_of_tracks:
            target_oh = torch.zeros(n)
            target_id = []
            targets = track["targets"]
            
            for ut in targets:
                ind = self.uuid_to_index[ut]
                target_oh[ind] = 1
                target_id.append(ind)
            
            if(len(target_id) > max_len):
                max_len = len(target_id)
                
            target_lst.append(target_oh)
            target_ind.append(target_id)
            
        for i, tt in enumerate(target_ind):
            while(len(tt) < max_len):
                tt.append(-1)
            target_ind[i]=tt

        return target_lst, target_ind


    def __len__(self):
        return len(self.all_indexs)

    def __getitem__(self, index):
   
        tmp_index = self.all_indexs[index]
        # flag = self.flip_tag[index]
        flag=False
        track = self.list_of_tracks[index]
        target = self.targets_ohe[index]
        target_ids = torch.Tensor(self.target_ind[index])
        if self.random:
            nl_idx = int(random.uniform(0, len(track["subjects"])-1))
            # print(len(track["subjects"]))
            # print(nl_idx)
            frame_idx = int(random.uniform(0, len(track["frames"])-1))
        else:
            nl_idx = 0
            frame_idx = 0
        text = track["subjects"][nl_idx]
        # if flag:
        #     text = text.replace("left","888888").replace("right","left").replace("888888","right")
        
        frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx])
        
        frame = default_loader(frame_path)
        box = track["boxes"][frame_idx]
        if self.crop_area == 1.6666667:
            box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
        else:
            box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))
        

        crop = frame.crop(box)
        if self.transform is not None:
            crop = self.transform(crop)

        if self.data_cfg.USE_MOTION:
            if self.index_to_uuid[tmp_index] in self.bk_dic:
                bk = self.bk_dic[self.index_to_uuid[tmp_index]]
            else:
                bk = default_loader(self.data_cfg.MOTION_PATH+"/%s.jpg"%self.index_to_uuid[tmp_index])
                self.bk_dic[self.index_to_uuid[tmp_index]] = bk
                bk = self.transform(bk)
                
            if flag:
                crop = torch.flip(crop,[1])
                bk = torch.flip(bk,[1])
            return crop,text,bk,target,target_ids,tmp_index
        if flag:
            crop = torch.flip(crop,[1])
        return crop,text,target,target_ids,tmp_index

#Need to modify for new usecase
class CityFlowNLInferenceDataset(Dataset):
    def __init__(self, data_cfg,transform = None):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg
        self.crop_area = data_cfg.CROP_AREA
        self.transform = transform
        with open(self.data_cfg.TEST_TRACKS_JSON_PATH) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.list_of_crops = list()
        for track_id_index,track in enumerate(self.list_of_tracks):
            for frame_idx, frame in enumerate(track["frames"]):
                frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, frame)
                box = track["boxes"][frame_idx]
                crop = {"frame": frame_path, "frames_id":frame_idx,"track_id": self.list_of_uuids[track_id_index], "box": box}
                self.list_of_crops.append(crop)
        self._logger = get_logger()

    def __len__(self):
        return len(self.list_of_crops)

    def __getitem__(self, index):
        track = self.list_of_crops[index]
        frame_path = track["frame"]

        frame = default_loader(frame_path)
        box = track["box"]
        if self.crop_area == 1.6666667:
            box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
        else:
            box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))
        

        crop = frame.crop(box)
        if self.transform is not None:
            crop = self.transform(crop)
        if self.data_cfg.USE_MOTION:
            bk = default_loader(self.data_cfg.MOTION_PATH+"/%s.jpg"%track["track_id"])
            bk = self.transform(bk)
            return crop,bk,track["track_id"],track["frames_id"]
        return crop,track["track_id"],track["frames_id"]

