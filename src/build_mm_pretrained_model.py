#!/usr/bin/env python3
#
#
#
import os
import copy
import argparse
from collections import OrderedDict
import torch, torchvision
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
#from detectron2.utils.visualizer import Visualizer, ColorMode
#from detectron2.data import MetadataCatalog, DatasetCatalog

#from detectron2.data.datasets import register_coco_instances

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.layers import *
from backbone import *


def load_backbone_weights(path):
    cp=torch.load(path)
    sd=OrderedDict()
    for k,v in cp['model'].items():
        if k.startswith('backbone.'):
            sd[k.replace('backbone.','')]=v

    return sd


def parse_options():
    parser=argparse.ArgumentParser(
                    description='build M3R-CNN pretrained model',
    )
    
    parser.add_argument(
        '-r','--rgb-weights',
        required=True,
         help='path to pretrained RGB model'
    )

    parser.add_argument(
        '-d','--depth-weights',
        required=True,
         help='path to pretrained depth model'
    )

    parser.add_argument(
        '-o','--output-path',
        required=True,
         help='path to outputting model'
    )

    return parser.parse_args()


opts=parse_options()

cfg=get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    )
)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.BACKBONE.FREEZE_AT=0

cfg.MODEL.BACKBONE.NAME='build_resnet_fpn_mm_backbone'
#
# set PIXEL_MEAN and PIXEL_STD as 4ch for data mapper
#
cfg.MODEL.PIXEL_MEAN=[0.5,0.5,0.5,0.357039]
cfg.MODEL.PIXEL_STD=[1.0,1.0,1.0,0.271628]
cfg.MODEL.WEIGHTS=None

cfg.OUTPUT_DIR=os.path.dirname(opts.output_path)


model=build_model(cfg)

cp=DetectionCheckpointer(model,cfg.OUTPUT_DIR)


model=cp.model
#
# load and store rgb backbone weights
#
model.backbone.rgb_bone.load_state_dict(
    load_backbone_weights(opts.rgb_weights)
)

#
# load and store depth backbone weights
#
model.backbone.depth_bone.load_state_dict(
    load_backbone_weights(opts.depth_weights)
)



cp.save(os.path.basename(opts.output_path))
