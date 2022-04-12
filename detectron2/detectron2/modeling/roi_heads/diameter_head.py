# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
import pdb
from torch.nn import functional as F
import cv2, os
import numpy as np

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from sklearn.metrics import mean_squared_error

__all__ = [
    "BaseDiamRCNNHead",
    "DiamRCNNConvUpsampleHead",
    "build_diam_head",
    "ROI_DIAMETER_HEAD_REGISTRY",
]


ROI_DIAMETER_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_DIAMETER_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@torch.jit.unused
def diameter_regr_loss(pred_diameter_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    
    # cls_agnostic_mask = pred_mask_logits.size(1) == 1
    # total_num_masks = pred_mask_logits.size(0)
    # mask_side_len = pred_mask_logits.size(2)
    # TODO reviure aquest assert
    # assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"
    

    # MAR gt and pred must have same format
    gt_classes = []
    gt_diameter = []
    
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        
        gt_diameter.append(instances_per_image.gt_diameter)
  
    if len(gt_diameter) == 0:
        return pred_diameter_logits.sum() * 0

    gt_diameter = cat(gt_diameter, dim=0)

    if True:
        pred_diameter_logits = pred_diameter_logits[:, 0]
        gt_diameter = gt_diameter[:,0]
    else:
        no_consequences_ = True
    
    # MAR compute the actual error
    if not gt_diameter.is_cuda:
        gt_diameter = gt_diameter.cuda()

    diameter_loss = F.mse_loss(gt_diameter, pred_diameter_logits)
    #diameter_loss = mean_squared_error(gt_diameter.cpu().detach().numpy(), pred_diameter_logits.cpu().detach().numpy())
    # podria implementar R2 measure com accuracy --> 1- rMSE

    return diameter_loss


def diameter_inference(pred_diam_logits: torch.Tensor, pred_instances: List[Instances]):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """

    
    for instances in  pred_instances:
        instances.pred_diameter = pred_diam_logits  # (1, Hmask, Wmask)


class BaseDiamRCNNHead(nn.Module):
    """
    Implement the basic Mask R-CNN losses and inference logic described in :paper:`Mask R-CNN`
    """

    @configurable
    def __init__(self, *, loss_weight: float = 1, vis_period: int = 0, output_dir: str = "/output", dataset_path: str = "/data"):
        """
        NOTE: this interface is experimental.

        Args:
            loss_weight (float): multiplier of the loss
            vis_period (int): visualization period
        """
        super().__init__()
        self.vis_period = vis_period
        self.loss_weight = loss_weight
        self.output_dir = output_dir
        self.dataset_path = dataset_path
        print("Vis period = " + str(self.vis_period))  #added by JGM
        print("Diammeter loss weight = " + str(self.loss_weight)) #added by JGM
        print("Output_dir = " + self.output_dir) #added by JGM
        print("Dataset_path = " + self.dataset_path) #added by JGM

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {"vis_period": cfg.VIS_PERIOD, "loss_weight": cfg.MODEL.ROI_DIAMETER_HEAD.DIAM_LOSS_WEIGHT, "output_dir": cfg.OUTPUT_DIR, "dataset_path": cfg.DATASET_PATH}

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """

        if not self.training:
            if len(x[:,0,0,0])>0:
                # 1. read current image names from current_images.txt
                with open(os.path.join(self.output_dir,'current_images_inference.txt')) as f:
                    lines = f.readlines()
                # 2. load corresponding depth maps
                # 3. add them to features tensor
                
                #depth_path = '/home/usuaris/imatge/jgene/multitask_RGBD/data/depthCropNpy'
                depth_path = os.path.join(self.dataset_path,'depthCropNpy')
                name = lines[0].split('.')[0]
                depth_map = np.load(os.path.join(depth_path,name+'.npy'))
                dim = len(x[0,0,0,:])
                depth_map = cv2.resize(depth_map, (dim,dim), interpolation=cv2.INTER_AREA)
                depth_map_tensor = torch.unsqueeze(torch.from_numpy(depth_map),dim=0)
                depth_map_tensor = depth_map_tensor.cuda()
            
                for x_ in range(len(x[:,0,0,0])):
                    # Dim: [1,257,a,a]
                    one_inst = torch.cat([torch.unsqueeze(x[x_,:,:,:],dim=0), torch.unsqueeze(depth_map_tensor,dim=0)],dim=1)
                    
                    if x_ == 0:
                        all_inst = one_inst
                    else:                    
                        all_inst = torch.cat([all_inst, one_inst],dim=0)
                
                x = all_inst
            else:
                dim = np.shape(x)[3]
                x = torch.empty(0, 257,dim,dim)
                x = x.cuda()


            
        x = self.layers(x)
        if self.training:
            return {"loss_diam": diameter_regr_loss(x, instances, self.vis_period) * self.loss_weight}
        else:
            diameter_inference(x, instances)
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError


# To get torchscript support, we make the head a subclass of `nn.Sequential`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
@ROI_DIAMETER_HEAD_REGISTRY.register()
class DiamRCNNConvUpsampleHead(BaseDiamRCNNHead, nn.Sequential):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of foreground classes (i.e. background is not
                included). 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []
        # MAR definir convs 
        cur_channels = input_shape.channels+1
        # MAR (he esborrat una conv si conv_dims[:-2])
        # MAR (he afegit una fonv si hi ha un if dins el for)
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("diam_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

            # if k+1 == len(conv_dims[:-1]):
            #     conv = Conv2d(
            #         cur_channels,
            #         conv_dim,
            #         kernel_size=3,
            #         stride=1,
            #         padding=1,
            #         bias=not conv_norm,
            #         norm=get_norm(conv_norm, conv_dim),
            #         activation=nn.ReLU(),
            #     )
            #     self.add_module("diam_fcn{}".format(k + 2), conv)
            #     self.conv_norm_relus.append(conv)
            #     cur_channels = conv_dim


        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        
        self.add_module("deconv_relu", nn.ReLU())
        cur_channels = conv_dims[-1]
        self.flatten = nn.Flatten()
        # self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)
        
        self.predictor = nn.Linear(cur_channels*28*28, 1)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    def layers(self, x):
        for layer in self:
            x = layer(x)
        return x


def build_diam_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    
    cfg.MODEL.ROI_DIAMETER_HEAD.NAME = "DiamRCNNConvUpsampleHead"
    name = cfg.MODEL.ROI_DIAMETER_HEAD.NAME
    return ROI_DIAMETER_HEAD_REGISTRY.get(name)(cfg, input_shape)
