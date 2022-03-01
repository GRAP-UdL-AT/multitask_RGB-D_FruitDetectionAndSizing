# codi https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=dq9GY37ml1kr
import matplotlib.pyplot as plt
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
import os,json,random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, GenericMask
from detectron2.data import MetadataCatalog, DatasetCatalog
import json
from detectron2.structures import BoxMode
import argparse
import utils_detection, utils_diameter
from detectron2.utils.visualizer import ColorMode
import metrics_detection, metrics_diameter
import pdb, argparse




def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate detection')
    parser.add_argument('--iou_thr',dest='iou_thr',default=0.5)
    parser.add_argument('--nms_thr',dest='nms_thr',default=0.1)
    parser.add_argument('--model_path',dest='model_path',default='../output/expnew_data_dec2021/model_0014225_bona.pth')
    parser.add_argument('--save_metrics',dest='save_metrics',default=False)
    parser.add_argument('--dataset_path',dest='dataset_path',default='/mnt/gpid08/users/mar.ferrer/data_fse/')
    parser.add_argument('--split',dest='split',default='test')
    parser.add_argument('--task',dest='task',default='detection',help='detection or diameter')
    parser.add_argument('--confs',dest='confs',default=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],help='confidences score to iterate over')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

# -----------------------------------------------------------------------------SET PARAMS--------------------------------------------------------------------------------------
    args = parse_args()
    iou_thr = args.iou_thr
    nms_thr = args.nms_thr
    model_path = args.model_path
    dataset_path = args.dataset_path
    split = args.split
    task = args.task
    p_list = []
    r_list = []
    f1_list = []
    ap_list = []
    conf_used = []

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print('LOADING_DATASET...')
    if not os.path.exists('dataset_dicts.npy'):
        dataset_dicts = utils_detection.get_FujiSfM_dicts(dataset_path,split)
        np.save('dataset_dicts.npy',np.array([dataset_dicts]))
    dataset_dicts = np.load('dataset_dicts.npy',allow_pickle=True)
    
    print('START')
    file = model_path.split('/')[-1]
    exp_name = model_path.split('/')[-2]
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 123.675]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0,1.0]
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thr
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

    d = 'val'
    DatasetCatalog.register("FujiSfM_" + d, lambda d=d: utils_detection.get_FujiSfM_dicts(dataset_path,d))
    MetadataCatalog.get("FujiSfM_" + d).set(thing_classes=["apple"])
    FujiSfM_metadata = MetadataCatalog.get("FujiSfM_"+d)
    #Inference & evaluation using the trained model
    cfg.MODEL.WEIGHTS = os.path.join(model_path)  # path to the model we just trained

    predictor = DefaultPredictor(cfg)

    from detectron2.structures import Boxes, BoxMode


    print('CALCULATING PRECISION, RECALL AND AP...')


    confidence_scores = args.confs
    if task == 'detection':   
        for k,s in enumerate(confidence_scores):

            P,R,F1,AP = metrics_detection.prec_rec_f1_ap(predictor,dataset_dicts,dataset_path+'images/'+split,s,FujiSfM_metadata,iou_thr) 

            p_list.append(P)
            r_list.append(R)
            f1_list.append(F1)
            ap_list.append(AP)
            conf_used.append(s)

        if args.save_metrics:
            np.save('array_f1.npy',np.array(f1_list))
            np.save('array_P.npy',np.array(p_list))
            np.save('array_R.npy',np.array(r_list))
            np.save('array_ap.npy',np.array(ap_list[0]))
    elif task == 'diameter':
        for k,s in enumerate(confidence_scores):
 
            metrics_diameter.prec_rec_f1_ap(predictor,dataset_dicts,dataset_path+'images/'+split,s,FujiSfM_metadata,split,iou_thr,args.save_metrics) 
 
    else:
        print('ERROR: THIS TASK IS NOT CONSIDERED: ',task)




                           
