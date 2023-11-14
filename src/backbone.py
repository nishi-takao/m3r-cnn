#
# M3R-CNN backbone module for detectron2
#
import torch
from torch import nn

from detectron2.config import get_cfg
from detectron2.layers import Conv2d,ShapeSpec
from detectron2.modeling.backbone import BACKBONE_REGISTRY,Backbone
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone

from collections import OrderedDict

__all__ = ["build_resnet_fpn_mm_backbone", "MMBackbone"]

class MMBackbone(Backbone):
    ################################################################
    #
    # Feature Mixing
    #
    class FeatureMixBlock(nn.Module):
        ################################################################
        #
        # Feature Mixing for Each Layer
        #
        class MixBlock(nn.Module):
            def __init__(self, in0_size, in1_size, out_size):
                super().__init__()
                block=[
                    # use detectron2.layers.Conv2d instead of nn.Conv2d
                    # to support empty inputs
                    Conv2d(
                        in0_size+in1_size,
                        out_size,
                        kernel_size=1
                    ),
                    nn.ReLU()
                ]
                
                self.block = nn.Sequential(*block)
                
            def forward(self, x0, x1):
                x=torch.cat([x0,x1],dim=1)
                return self.block(x)
        #
        # end of MixBlock
        #
        ################################################################
        
        def __init__(self,backbone_layers,channels):
            super().__init__()
            
            d=OrderedDict()
            self._backbone_layers=backbone_layers
            for k in self._backbone_layers:
                d[k]=self.__class__.MixBlock(
                    channels,
                    channels,
                    channels
                )
            
            self.block=nn.ModuleDict()
            self.block.update(d)
        
        def forward(self, x0, x1):
            mixed_features = OrderedDict()
            for k in self._backbone_layers:
                mixed_features[k] = self.block[k](
                    x0[k],
                    x1[k]
                )
                
            return mixed_features
    #
    # end of FeatureMixBlock
    #
    ################################################################
    
    def __init__(self, cfg, input_shape):
        super().__init__()
        
        _cfg=get_cfg()
        _cfg.merge_from_other_cfg(cfg)
        _cfg.MODEL.BACKBONE.NAME='build_resnet_fpn_backbone'
        
        self.rgb_bone=build_resnet_fpn_backbone(_cfg,ShapeSpec(channels=3))
        self.depth_bone=build_resnet_fpn_backbone(_cfg,ShapeSpec(channels=1))
        self.feature_mix=self.__class__.FeatureMixBlock(
            _cfg.MODEL.RPN.IN_FEATURES,
            _cfg.MODEL.FPN.OUT_CHANNELS
        )

        #
        # fix me
        #
        self._plane_rgb=slice(0,3)
        self._plane_depth=slice(3,None)

    @property
    def size_divisibility(self):
        return self.rgb_bone.size_divisibility

    @property
    def padding_constraints(self):
        return self.rgb_bone.padding_constraints
    
    def output_shape(self):
        return self.rgb_bone.output_shape()
    
    def forward(self,x):
        """
        Args:
          x: Tensor of shape (N,C,H,W). C shold be 4: [R,G,B,D]
        
        Returns:
            dict[str->Tensor]: Same as FPN backbone
        """
        x0=self.rgb_bone(x[:,self._plane_rgb,:,:])
        x1=self.depth_bone(x[:,self._plane_depth,:,:])
        
        return self.feature_mix(x0,x1)

@BACKBONE_REGISTRY.register()
def build_resnet_fpn_mm_backbone(cfg, input_shape: ShapeSpec):
    return MMBackbone(cfg,input_shape)
