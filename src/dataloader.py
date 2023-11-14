#
# 
#
import copy
import logging
import numpy as np
import os
from typing import List, Optional, Union

import torch, torchvision

from detectron2.config import configurable
from detectron2.data.dataset_mapper import *
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import \
    DatasetMapper,\
    MetadataCatalog,\
    build_detection_train_loader,\
    build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.engine import DefaultTrainer,DefaultPredictor
from detectron2.modeling import build_model

from evaluator import COCOEvaluator,LossEvalHook


####################################################################
#
# NPZ formated RGBD data loader
#
# DEPTH mean and std are 0.357039 0.271628
class RGBDNPZLoader(DefaultTrainer):

    ####################################################################
    #
    # 4ch NPZ (h,w,4) Data mapper
    #
    class Mapper(DatasetMapper):
        @configurable
        def __init__(
                self,
                is_train: bool,
                *,
                augmentations: List[Union[T.Augmentation, T.Transform]],
                image_format: str,
                use_instance_mask: bool = False,
                use_keypoint: bool = False,
                instance_mask_format: str = "polygon",
                keypoint_hflip_indices: Optional[np.ndarray] = None,
                precomputed_proposal_topk: Optional[int] = None,
                recompute_boxes: bool = False,
                #
                # input channel numvber
                # slice(0,3) #=> RGB,
                # slice(3,None) #=> Depth,
                # slice(0,None) #=> RGBD
                in_channels: slice = slice(0,None), 
        ):
            super().__init__(
                is_train,
                augmentations=augmentations,
                image_format=image_format,
                use_instance_mask=use_instance_mask,
                use_keypoint=use_keypoint,
                instance_mask_format=instance_mask_format,
                keypoint_hflip_indices=keypoint_hflip_indices,
                precomputed_proposal_topk=precomputed_proposal_topk,
                recompute_boxes=recompute_boxes
            )
            self._in_channels=in_channels
        
        @classmethod
        def from_config(cls, cfg, is_train: bool = True):
            ret=super().from_config(cfg, is_train)
            
            l=len(cfg.MODEL.PIXEL_MEAN)
            if(l==1):
                ret['in_channels']=slice(3,None)
            elif(l==3):
                ret['in_channels']=slice(0,3)
            else:
                ret['in_channels']=slice(0,None)
            
            return ret
    
        def __call__(self, dataset_dict):
            dataset_dict = copy.deepcopy(dataset_dict)
            with open(dataset_dict["file_name"],'rb') as f:
                image=np.load(f)['arr_0'][
                    :,:,self._in_channels
                ] # (H,W,C)
            
            sem_seg_gt = None
            
            aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
            transforms = self.augmentations(aug_input)
            image, sem_seg_gt = aug_input.image, aug_input.sem_seg
            
            image_shape = image.shape[:2]  # h, w
            dataset_dict["image"] = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            )
            
            if not self.is_train:
                dataset_dict.pop("annotations", None)
                dataset_dict.pop("sem_seg_file_name", None)
                return dataset_dict
            
            if "annotations" in dataset_dict:
                self._transform_annotations(
                    dataset_dict,
                    transforms,
                    image_shape
                )
            
            return dataset_dict

    #
    # end of Mapper
    #
    ####################################################################

    ####################################################################
    #
    # 4ch NPZ (h,w,4) Predictor
    # suppress format check
    #
    class Predictor(DefaultPredictor):
        def __init__(self, cfg):
            self.cfg = cfg.clone()  # cfg can be modified by model
            self.model = build_model(self.cfg)
            self.model.eval()
            if len(cfg.DATASETS.TEST):
                self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
            
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            
            self.aug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
                cfg.INPUT.MAX_SIZE_TEST
            )
            
        def __call__(self, original_image):
            with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
                # Apply pre-processing to image.
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(
                    original_image
                ).apply_image(original_image)
                image = torch.as_tensor(
                    image.astype("float32").transpose(2, 0, 1)
                )
                image.to(self.cfg.MODEL.DEVICE)
            
            inputs = {"image": image, "height": height, "width": width}
            
            predictions = self.model([inputs])[0]
            return predictions
    #
    # end of Predictor
    #
    ####################################################################
        
    @classmethod
    def build_train_loader(cls,cfg):
        mapper=cls.Mapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls,cfg,dataset_name):
        mapper=cls.Mapper(cfg,is_train=False)
        return build_detection_test_loader(cfg,dataset_name,mapper=mapper)
    
    @classmethod
    def build_evaluator(
            cls,
            cfg,
            dataset_name,
            output_folder=None,
            distributed=True
    ):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name,cfg,distributed,output_folder)
    
    def build_hooks(self):
        hooks=super().build_hooks()
        mapper=cls.Mapper(cfg,is_train=False)
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            self.__class__.build_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0]
            )
        ))
        return hooks

#
# end of RGBDNPZLoader
#
####################################################################
