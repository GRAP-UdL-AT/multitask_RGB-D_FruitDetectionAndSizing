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
import eval_metrics
import pdb, argparse




def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate detection')
    parser.add_argument('--iou_thr',dest='iou_thr',default=0.5)
    parser.add_argument('--nms_thr',dest='nms_thr',default=0.1)
    parser.add_argument('--model_path',dest='model_path',default='./output/trial30_DLW_5_bs4_gpu2_t04ext/model_0002999.pth')
    parser.add_argument('--save_metrics',dest='save_metrics',default=True)
    parser.add_argument('--dataset_path',dest='dataset_path',default='/mnt/gpid07/users/jordi.gene/multitask_RGBD/data/')
    parser.add_argument('--split',dest='split',default='test')
    parser.add_argument('--test_name',dest='test_name',default='eval_00')
    parser.add_argument('--year',dest='year',default='all')
    #parser.add_argument('--confs',dest='confs',default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',help='confidences score to iterate over')
    parser.add_argument('--confs',dest='confs',default='0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99',help='confidences score to iterate over')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

# -----------------------------------------------------------------------------SET PARAMS--------------------------------------------------------------------------------------
    args = parse_args()
    iou_thr = float(args.iou_thr)
    nms_thr = float(args.nms_thr)
    model_path = args.model_path
    save_metrics = args.save_metrics
    dataset_path = args.dataset_path
    split = args.split
    test_name = args.test_name
    year = args.year
    confidence_scores = [float(i) for i in args.confs.split(',')]
    p_list = []
    r_list = []
    f1_list = []
    ap_list = []
    mae_list = []
    conf_used = []

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print('LOADING_DATASET...')
    dataset_dicts_file = os.path.join(dataset_path,split+'_dataset_dicts.npy')
    if not os.path.exists(dataset_dicts_file):
        dataset_dicts = utils_detection.get_FujiSfM_dicts(dataset_path,split)
        np.save(dataset_dicts_file,np.array([dataset_dicts]))
    dataset_dicts = np.load(dataset_dicts_file,allow_pickle=True)
    
    print('START')
    file = model_path.split('/')[-1]
    exp_name = model_path.split('/')[-2]
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 1.64]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0,1.42]
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thr
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.DATASET_PATH = dataset_path
    cfg.OUTPUT_DIR="./output/"+str(exp_name)+"/"+test_name
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # d = 'val'
    d = split   #by JGM
    DatasetCatalog.register("FujiSfM_" + d, lambda d=d: utils_detection.get_FujiSfM_dicts(dataset_path,d))
    MetadataCatalog.get("FujiSfM_" + d).set(thing_classes=["apple"])
    FujiSfM_metadata = MetadataCatalog.get("FujiSfM_"+d)
    #Inference & evaluation using the trained model
    cfg.MODEL.WEIGHTS = os.path.join(model_path)  # path to the model we just trained

    predictor = DefaultPredictor(cfg)

    from detectron2.structures import Boxes, BoxMode


    print('CALCULATING PRECISION, RECALL AND AP...')

    P,R,F1,AP,MAE = eval_metrics.prec_rec_f1_ap_MAE(predictor,dataset_dicts,dataset_path+'images/'+split,confidence_scores,split,iou_thr,save_metrics, cfg.OUTPUT_DIR, year)
                           
