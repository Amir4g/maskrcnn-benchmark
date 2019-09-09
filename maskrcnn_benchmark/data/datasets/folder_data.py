import os
from PIL import Image
import numpy as np
import json
import logging

import torch
import torchvision
#from .coco import coco
#from maskrcnn_benchmark.data.datasets.coco import COCODataset #as coco
from maskrcnn_benchmark.structures.bounding_box import BoxList
#from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
#from maskrcnn_benchmark.structures.segmentation_mask import Mask, MaskList
import torch.utils.data as data
import copy

class FOLDER_DATA(object):
    """
    target (Bboxlist) has fields: 
        [
        'instance_ids',: decode from original mask
        'category_id', : object classes id
        'labels', : object classes
        'bin_mask_list',: a list of binary mask
        'instance_mask': original mask loade from Annotations
        ]
    __getitem__ will return sample:
        img, target, idx, meta = sample 
        mask = target.get_field('instance_mask') # in shape [1, H, W]
    """
    def __init__(self, ann_file,  
            db_root_dir, transforms=None, cache_data=True):
        self.db_root_dir = db_root_dir
        self.ann_file = ann_file

        self.transforms_davis = transforms
        logger = logging.getLogger(__name__)
        logger.info('init datasets: %s'%db_root_dir)
        self.palette = None
        print(db_root_dir, self.ann_file)
        if 'txt' in self.ann_file:
            with open(os.path.join(self.ann_file)) as f:
                seqs = f.readlines()
            if 'track' not in db_root_dir:
                seqs = [seq.rstrip("\n").strip() for seq in seqs]
            else:
                seqs = [seq.rstrip("\n").strip()[1:] for seq in seqs]
        elif 'json' in self.ann_file:
            seqs = json.load(open(self.ann_file, 'r')) 
            if 'videos' in seqs: # and 'meta' in self.ann_file:
                # loading ytb/meta.json 
                seqs = list(seqs['videos'].keys()) 
            else:
                seqs = list(seqs.keys())

        self.DATA = [] 
        self.vid_fid_2_index = {}
        self.seqs = seqs
        logger.info('start loading data..')
        count = 0
        for _video in self.seqs:
            self.vid_fid_2_index[_video] = {}
            frames = np.sort(os.listdir(db_root_dir+ '/' + _video))
            for f in frames:
                self.DATA.append({'video': _video, 'frame': f})

                self.vid_fid_2_index[_video][f.split('.')[0]] = count # also generate a DATA-INDEX 
                count += 1
        if 'json' in self.ann_file:
           indexfile = self.ann_file.replace('.json', '-CACHE_maskben_folderdata_index_%d.json'%len(self.seqs)).split('/')[-1] 
           indexfile = 'data/folder_data/%s'%indexfile
           logger.info('save indexfile at %s'%indexfile)
           json.dump({'index2vidfid':self.DATA, 'vidfid2index':self.vid_fid_2_index}, open(indexfile, 'w'))

        logger.info('vid %s .. total:  %d'%(_video[0], count)) 
        logger.info('done')
    def __len__(self):
        return len(self.DATA)

    def __getitem__(self, idx):
        """
        return loaded image and target (BBoxList):
            add field: "instance_ids", "category_id", "annotation_path", "labels",
                        "bin_mask_list", "instance_mask"
        """
        sample = self.DATA[idx]
        image_path = sample['video'] +'/' + sample['frame']
        img = Image.open(self.db_root_dir + '/' +image_path).convert("RGB") 
        i = torch.from_numpy(np.array(img))
        self.DATA[idx]['image_width'] = np.array(i.shape[1])
        self.DATA[idx]['image_height'] = np.array(i.shape[0])

        #meta = sample['meta']
        #mask_path = sample['mask_path']
        # mpath = os.path.join(self.db_root_dir, mask_path)
        
        target = img
        if self.transforms_davis is not None:
            img, target = self.transforms_davis(img, target)
    
        return img, target, idx #, meta
    
    # ---------------------------------------
    # add helper function for inference.py
    # ---------------------------------------
    def get_img_info(self, index):
        meta = self.DATA[index] #['meta']
        if 'image_width' not in meta:
            self.__getitem__(index)
        return {
                "width": meta["image_width"], 
                "height": meta["image_height"],
                "seq": meta["video"],
                "frame": meta["frame"]
                }


    def get_groundtruth(self, idx):
        img, target, idx = self.__getitem__(idx)
        return target


