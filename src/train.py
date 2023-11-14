#!/usr/bin/env python3
#
#
#
import sys, os, math #, json, cv2, random
import copy
import argparse
import numpy as np
import re
import torch, torchvision

import logging

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import default_argument_parser,hooks,launch
#from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.layers import *
from detectron2.checkpoint.catalog import *

from backbone import *
from dataloader import RGBDNPZLoader

DATASET_TRAIN_NAME='targets_train'
DATASET_VAL_NAME='targets_val'

#
# To distinguish between explicit and non-explicit options,
# instead of setting 'default' in add_argument(),
# define defaults here and call fill_defaults() as needed.
#
OPTS_DEFAULTS={
    'batch_size':32,
    'epochs':30,
    'lr':0.001,
    'lr_gamma':0.1,
    'lr_steps':None,
    'max_image_size':1333,
    'freeze_at':0
}

def fill_defaults(opts):
    for k,v in OPTS_DEFAULTS.items():
        if(opts.__getattribute__(k) is None):
            opts.__setattr__(k,v)
    
    return opts
    
def parse_options():
    parser=default_argument_parser()

    # chaneg num-gpus options default value from 1 to -1
    i=-1
    for j,o in enumerate(parser._actions):
        if(o.dest=='num_gpus'):
            i=j
            break
    if(i>=0):
        parser._actions[i].default=-1
        
    parser.add_argument(
        "-t","--train_data",
        default='../dataset/train/',
        help="training data dir"
    )
    parser.add_argument(
        "-v","--val_data",
        default=None,
        help="validation data dir. If not specified, "\
        "validation data will be sampled from training data."
    )
    parser.add_argument(
        "-o","--output_dir",
        default='../model',
        help="models storing dir"
    )
    parser.add_argument(
        "--mode",
        choices=['mm','rgb', 'depth'],
        default='mm',
        help="training mode"
    )
    parser.add_argument(
        "-b","--batch_size",
        type=int,
        help='batch size'
    )
    parser.add_argument(
        '-e',"--epochs",
        type=int,
        help='number of epoch'
    )
    parser.add_argument(
        "--lr",
        type=float,
        help='learning rate'
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        help='learning rate step down gamma'
    )
    parser.add_argument(
        "--lr_steps",
        nargs="*",
        help='learning rate step down points (E: itr/epoch, I: total itr)'
    )

    parser.add_argument(
        "--max_image_size",
        type=int,
        help='maximum image size'
    )

    parser.add_argument(
        "--freeze_at",
        type=int,
        help='backbone freeze layer'
    )

    parser.add_argument(
        "-w","--pretrain_weights",
        default=None,
        help='pre-trained backbone weights'
    )
    
    parser.add_argument(
        "--check_point",
        default=None,
        help='checkpoint to resume'
    )
    
    parser.add_argument(
        "--save_config",
        action='store_true',
        help="Save current configuration data"
    )
    
    return parser.parse_args()
    
def parse_lr_steps(stepstr,itr_per_epoch,max_itr):
    if(stepstr is None):
        return ()
    
    r=re.compile('^([\.\d]+)([eEiI]?)$')
    buf=[]
    for s in stepstr:
        m=r.match(s)
        if(m is not None):
            if(m[2]==''):
                buf.append(int(m[1]))
            else:
                x=float(m[1])
                u=m[2].lower()
                if(u=='e'):
                    buf.append(int(x*itr_per_epoch))
                elif(u=='i'):
                    buf.append(int(x*max_itr))
    
    return tuple(buf)

def register_datasets(opts,train_only=False):
    register_coco_instances(
        DATASET_TRAIN_NAME,
        {},
        opts.train_data,
        os.path.dirname(opts.train_data)
    )
    
    if((opts.val_data is not None) and (not train_only)):
        register_coco_instances(
            DATASET_VAL_NAME,
            {},
            opts.val_data,
            os.path.dirname(opts.val_data)
        )


def setup():
    opts=parse_options()
    
    cfg=get_cfg()

    if(opts.config_file is None):
        opts=fill_defaults(opts)
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
            )
        )
    else:
        cfg.merge_from_file(opts.config_file)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES=2
    
    cfg.DATASETS.TRAIN = (DATASET_TRAIN_NAME,)
    if(opts.val_data is None):
        cfg.DATASETS.TEST = ()
    else:
        cfg.DATASETS.TEST=(DATASET_VAL_NAME,)
    
    cfg.OUTPUT_DIR=opts.output_dir
    
    #
    # DEPTH mean and std are 0.357039 0.271628
    # However, we disable pixel standardizing proc.  
    #
    cfg.MODEL.PIXEL_MEAN=[0.,0.,0.,0.] #[0.5,0.5,0.5,0.357039]
    cfg.MODEL.PIXEL_STD=[1.,1.,1.,1.]  #[1.0,1.0,1.0,0.271628]
    if(opts.mode=='rgb'):
        # RGB 3ch mode
        cfg.MODEL.PIXEL_MEAN=cfg.MODEL.PIXEL_MEAN[0:3]
        cfg.MODEL.PIXEL_STD=cfg.MODEL.PIXEL_STD[0:3]
    elif(opts.mode=='depth'):
        # Depth 1ch mode
        cfg.MODEL.PIXEL_MEAN=cfg.MODEL.PIXEL_MEAN[3:]
        cfg.MODEL.PIXEL_STD=cfg.MODEL.PIXEL_STD[3:]
        cfg.MODEL.WEIGHTS=None
    else:
        # RGB 3ch + Depth 1ch
        cfg.MODEL.BACKBONE.NAME='build_resnet_fpn_mm_backbone'
        cfg.MODEL.WEIGHTS=None

    if(opts.batch_size is not None):
        cfg.SOLVER.IMS_PER_BATCH=opts.batch_size

    if(opts.freeze_at is not None):
        cfg.MODEL.BACKBONE.FREEZE_AT=0 if opts.freeze_at<=0 else opts.freeze_at
    
    if(torch.cuda.is_available()):
        if(opts.num_gpus<0):
            opts.num_gpus=torch.cuda.device_count()
    else:
        opts.num_gpus=0
    
    register_datasets(opts,True)
    n_images=len(DatasetCatalog.get(DATASET_TRAIN_NAME))
    itr_per_epoch=math.ceil(n_images/cfg.SOLVER.IMS_PER_BATCH)

    sys.stderr.write(
        '** (n_images, IMS_PER_BATCH, itr/epochs) = (%d, %d, %d)\n'%(
            n_images,cfg.SOLVER.IMS_PER_BATCH,itr_per_epoch
        )
    )

    if((opts.epochs is not None) and (opts.epochs>0)):
        # https://stackoverflow.com/questions/63578040/how-many-images-per-iteration-in-detectron2
        cfg.SOLVER.MAX_ITER=int(itr_per_epoch*opts.epochs)
        # save checkpoint for each epoch
        cfg.SOLVER.CHECKPOINT_PERIOD=itr_per_epoch
        sys.stderr.write(
            '** epochs = (%d)\n'%(opts.epochs)
        )
    
    sys.stderr.write(
        '** cfg.SOLVER.MAX_ITER = (%d)\n'%(cfg.SOLVER.MAX_ITER)
    )

    cfg.TEST.EVAL_PERIOD=0 if opts.val_data is None else itr_per_epoc

    if(opts.lr is not None):
        cfg.SOLVER.BASE_LR=opts.lr
        
    if(opts.lr_gamma is not None):
        cfg.SOLVER.GAMMA=opts.lr_gamma

    if(opts.lr_steps is not None):
        cfg.SOLVER.STEPS=parse_lr_steps(
            opts.lr_steps,
            itr_per_epoch,
            cfg.SOLVER.MAX_ITER
        )

    sys.stderr.write(
        '** cfg.SOLVER.STEPS = (%s)\n'%(str(cfg.SOLVER.STEPS))
    )

    if(opts.max_image_size is not None):
        cfg.INPUT.MAX_SIZE_TRAIN=opts.max_image_size
        cfg.INPUT.MAX_SIZE_TEST=opts.max_image_size

    if(opts.check_point is not None):
        cfg.MODEL.WEIGHTS=opts.check_point
        opts.resume=True
    elif(opts.pretrain_weights is not None):
        cfg.MODEL.WEIGHTS=opts.pretrain_weights
        opts.resume=False

    #
    # save current config
    #
    if(opts.save_config):
        path=os.path.join(
            cfg.OUTPUT_DIR,
            'config.yaml'
        )
        with open(path,'w') as f:
            f.write(cfg.dump())
    
    #
    # copy command line options to config
    #
    cfg['_LOCAL_']=detectron2.config.CfgNode({
        'train_data': opts.train_data,
        'val_data': opts.val_data,
        'num_gpus':opts.num_gpus,
        'num_machines':opts.num_machines,
        'machine_rank':opts.machine_rank,
        'dist_url':opts.dist_url,
        'eval_only':opts.eval_only,
        'resume':opts.resume
    })
    
    return cfg

def main(cfg):
    opts=copy.deepcopy(cfg._LOCAL_)
    cfg.__delitem__('_LOCAL_')
    
    register_datasets(opts)
    
    if opts.eval_only:
        '''
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
        '''
        raise NotImplementedError('fixme')
    
    trainer=RGBDNPZLoader(cfg)
    trainer.resume_or_load(resume=opts.resume)
    if cfg.TEST.AUG.ENABLED:
        '''
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
        '''
        raise NotImplementedError('fixme')
        
    return trainer.train()

if __name__ == "__main__":
    cfg=setup()
    
    launch(
        main,
        cfg._LOCAL_.num_gpus,
        num_machines=cfg._LOCAL_.num_machines,
        machine_rank=cfg._LOCAL_.machine_rank,
        dist_url=cfg._LOCAL_.dist_url,
        args=(cfg,),
    )
