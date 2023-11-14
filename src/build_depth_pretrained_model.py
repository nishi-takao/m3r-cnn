#!/usr/bin/env python3
#
#
#
import os
import copy
import argparse
import torch, torchvision
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.layers import *

def parse_options():
    DEPTH_PRETRAINED_MODEL_BASENAME='R-101-1ch'
    
    parser=argparse.ArgumentParser(
                    description='build depth (1ch) pretrained model',
    )

    parser.add_argument(
        '-o','--output-path',
        default=os.path.join(
            os.path.dirname(__file__),
            '../model/pretrain',
            DEPTH_PRETRAINED_MODEL_BASENAME
        ),
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

cfg.OUTPUT_DIR=os.path.dirname(opts.output_path)


model=build_model(cfg)
cp=DetectionCheckpointer(model,cfg.OUTPUT_DIR)
cp.resume_or_load(cfg.MODEL.WEIGHTS,resume=False)

model=cp.model
device=copy.deepcopy(model.device)
if(device.type!='cpu'):
    model.cpu()

backbone_conv1=model.backbone.bottom_up.stem.conv1
backbone_conv1_weight=\
    copy.deepcopy(backbone_conv1.weight)[:,1,:,:].unsqueeze(1)

model.backbone.bottom_up.stem.conv1=Conv2d(
    1,
    64,
    kernel_size=(7, 7),
    stride=(2, 2),
    padding=(3, 3),
    bias=False,
    norm=FrozenBatchNorm2d(num_features=64),
)
model.backbone.bottom_up.stem.conv1.weight=torch.nn.Parameter(
    backbone_conv1_weight
)
if(device.type!='cpu'):
    model.cuda(device.index)

cp.save(os.path.basename(opts.output_path))
