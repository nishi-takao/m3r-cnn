#!/usr/bin/env python3
#
#
import os
import argparse
import json
import logging
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import inference_on_dataset
from backbone import *
from dataloader import RGBDNPZLoader


DATASET_TEST_NAME='targets_test'
DEFAULT_CONFIG_FILE='config.yaml'
DEFAULT_WEIGHTS_FILE='model_final.pth'
RESULT_JSON_FILE='ap.json'

def parse_options():
    parser= argparse.ArgumentParser(
        description='m3rcnn evaluator',
    )

    parser.add_argument(
        "-t","--test_data",
        default='../dataset/test/',
        help="test data dir"
    )
    parser.add_argument(
        "-o","--output_dir",
        default='../result',
        help="test results storing dir"
    )

    parser.add_argument(
        "-m","--model_dir",
        default=None,
        help='directory for model config and weights'
    )
    
    parser.add_argument(
        "--config",
        default=None,
        help='model configuration file'
    )
    parser.add_argument(
        "--weight",
        default=None,
        help='model weights'
    )
    
    return parser


argarser=parse_options()
opts=argarser.parse_args()

#load the config file, configure the threshold value, load weights 
cfg = get_cfg()
cf=opts.config
if(cf is None):
    if(opts.model_dir is None):
        argarser.error('model_dir or config and weights option is required')
    
    cf=os.path.join(opts.model_dir,DEFAULT_CONFIG_FILE)

cfg.merge_from_file(cf)

cfg.OUTPUT_DIR=opts.output_dir

#register your data
register_coco_instances(
    DATASET_TEST_NAME,
    {},
    opts.test_data,
    os.path.dirname(opts.test_data)
)
cfg.DATASETS.TEST=(DATASET_TEST_NAME,)


cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

wf=opts.weight
if(wf is None):
    if(opts.model_dir is None):
        argarser.error('model_dir or config and weights option is required')

    wf=os.path.join(opts.model_dir,DEFAULT_WEIGHTS_FILE)

cfg.MODEL.WEIGHTS = wf


predictor=RGBDNPZLoader.Predictor(cfg)
val_loader=RGBDNPZLoader.build_test_loader(
    cfg,
    DATASET_TEST_NAME,
)
evaluator=RGBDNPZLoader.build_evaluator(
    cfg,
    DATASET_TEST_NAME,
    output_folder=cfg.OUTPUT_DIR,
    distributed=False
)

#Use the created predicted model in the previous step
results=inference_on_dataset(
    predictor.model,
    val_loader,
    evaluator
)

with open(os.path.join(cfg.OUTPUT_DIR,RESULT_JSON_FILE),'w') as f:
    json.dump(results,f,indent=2)
