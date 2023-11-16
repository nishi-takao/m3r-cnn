# M3R-CNN on detectron2

NISHI, Takao <nishi.t.es@osaka-u.ac.jp>

## What's this?

[M3R-CNN](https://doi.org/10.1080/01691864.2023.2257266) (Multi-modal Mask R-CNN) is a network for the instance-segmentation task that takes RGB and depth as input and obtains high generalizability with small training data.

This is an implementation of M3R-CNN on detectron2.


## Installing and Running

### Dependencies

* Numpy
* [PyTorch](https://github.com/pytorch/pytorch)
* [detectron2](https://github.com/facebookresearch/detectron2)

### Datasets

This repository contains no datasets.
You will need to prepare your own RGB+D dataset for your own use.

The dataset must contain annotations in COCO format and scene data in npz format. Each scene data must be an RGBD 4ch numpy array dumped in npz format.

Note that the loaded scene data are not normalized except for image size.

### Training

The network is trained in three stages: RGB backbone, Depth backbone, and fine-tuning.

#### Trainig RGB backbone

Set the mode option to 'rgb' and let it train about 15 epochs.
A pre-trained model in ImageNet is automatically applied.

```
$ cd src/
$ ./train.py --mode rgb \
             -t PATH/TO/YOUR/TRINING_DATASET/JSON.FILE \
             -o PATH/TO/TRINED/RGB_BACKBONE/ \  # output directory
             -e 15
```

If the weights become NaN or Inf during training, try to freeze up to the second layer by adding the '--freeze_at 2' option.

#### Trainig Depth backbone

First, the RGB 3-channel ImageNet pre-training model is converted to 1-channel.

```
$ cd src/
$ ./build_depth_pretrained_model.py -o PATH/TO/DEPTH_PRETRAINED_MODEL # output file
```

Then, using this pre-training model, we train about 15 epochs with 'depth' as the mode.
```
$ ./train.py --mode depth \
             -t PATH/TO/YOUR/TRINING_DATASET/JSON.FILE \
             -w PATH/TO/DEPTH_PRETRAINED_MODEL \
             -o PATH/TO/TRINED/DEPTH_BACKBONE/ \ # output directory
             -e 15
```

#### Fine-tuning

First, the RGB and Depth training results are integrated.

```
$ cd src/
$ ./build_mm_pretrained_model.py -r PATH/TO/TRINED/RGB_BACKBONE \
                                 -d PATH/TO/TRINED/DEPTH_BACKBONE \
                                 -o PATH/TO/MM_BACKBONE # output file
```

Then, using this pre-training model, we train about 15 epochs with 'mm' as the mode.
It is recommended to set the learning rate 'lr' to be about ten times the RGB and Depth (default value is 0.001) and to decay to 1/10 at 2.3 epochs and 11.5 epochs, respectively.

To output the necessary config file during the evaluation, please also specify the '--save_config' option.

```
$ ./train.py --mode mm \
             -t PATH/TO/YOUR/TRINING_DATASET/JSON.FILE \
             -w PATH/TO/MM_BACKBONE \
             -o PATH/TO/OUTPUT/TRINED/MM/ \ # output directory
             -e 15 \
             --lr 0.01 \
             --lr_gamma 0.1 \
             --lr_steps 2.3E 11.5E \
             --save_config
```

#### Notes
The default values for batch size '--bs=32' and learning rate ('--lr=0.001') options are set based on the assumption that training is performed on a maximum 1333 x 1000px image in an environment with two RTX6000A units (48GB x2 VRAM).
These values should be adjusted to match the VRAM capacity.
The maximum image size can be specified with the '--max_image_size' option.

If you need validation, specify the validation dataset with the '-v' option.
For each epoch (number of iterations obtained by number of images of training data / batch size), losses and AP calculations are performed on the validation data.

By default, the training script uses all GPUs for distributed training; if you want to limit the number of GPUs, use the '--num-gpus' option. If this option is set to 0, no GPUs are used.


### Evaluation
Run test.py specifying the directory containing the trained model (and its config file) and the test dataset.

```
$ cd src/
$ ./test.py -t PATH/TO/YOUR/TEST_DATASET/JSON.FILE \
            -o PATH/TO/DIRECTORY/TO_STORE_RESULTS_FILES/ \
            -m PATH/TO/DIR$ECTORY/TRINED/MM/
```

## License
BSD 3-Clause License


## Reference
```
@article{nishi2023m3r,
  title={M3R-CNN: on effective multi-modal fusion of RGB and depth cues for instance segmentation in bin-picking},
  author={Nishi, Takao and Kawasaki, Shinya and Iewaki, Kosuke and Okura, Fumio and Petit, Damien and Takano, Yoichi and Harada, Kensuke},
  journal={Advanced Robotics},
  volume={37},
  number={18},
  pages={1143--1157},
  year={2023},
  publisher={Taylor \& Francis}
}
```
