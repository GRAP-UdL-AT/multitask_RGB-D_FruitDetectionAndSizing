# codi https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=dq9GY37ml1kr

import torch, torchvision
# Some basic setup:
# Setup detectron2 logger
import detectron2
import pdb
from detectron2.utils.logger import setup_logger
setup_logger()
import pickle5 as pickle

# import some common libraries
import numpy as np
import os, json, cv2, random,argparse
import utils_detection

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import wandb
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils_detectron
import detectron2.data.transforms as T
import copy
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import logging




def mapper(dataset_dict):

    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # can use other ways to read image
    image_bgr = utils_detectron.read_image(dataset_dict["file_name"], format="BGR")
    if not os.path.exists(dataset_dict['depth_file']):
        print('NO ESISTE',dataset_dict['depth_file'])
 
    depth_map = np.load(dataset_dict["depth_file"])
    try:
        depth_map = cv2.resize(depth_map, (np.shape(image_bgr[:,:,0])[1],np.shape(image_bgr[:,:,0])[0]), interpolation=cv2.INTER_AREA)
    
    except:
        raise Exception("COULD NOT RESIZE"+dataset_dict['depth_file'])

    image = np.zeros((np.shape(image_bgr)[0],np.shape(image_bgr)[1],4))
    image[:,:,0:3] = image_bgr
    image[:,:,3] = depth_map
    # See "Data Augmentation" tutorial for details usage
    augs = T.AugmentationList([
        T.RandomFlip(prob=0.5),
        T.RandomBrightness(0.9, 1.1)
    ])
    auginput = T.AugInput(image)
    transform = augs(auginput)
    image = torch.from_numpy(auginput.image.transpose(2, 0, 1).copy())
    
    annos = [
        utils_detectron.transform_instance_annotations(annotation, [transform], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]
    return {
        # create the format that the model expects
        "filename":dataset_dict['file_name'],
        "depth_file":depth_map,
        "image_id":dataset_dict['image_id'],
        "height": dataset_dict['height'],
        "width": dataset_dict['width'],
        "image": image,  
        "instances": utils_detectron.annotations_to_instances(annos, image.shape[1:])
    }


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        #pdb.set_trace()
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        diam_losses = []
        for idx, inputs in enumerate(self._data_loader): 
            with open('current_images.txt', 'w') as f:
                f.write("%s\n" % inputs[0]['filename'])           
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch, diam_loss_batch = self._get_loss(inputs) #added by JGM(add diam_loss_batch to be plotted in tensorboard)
            losses.append(loss_batch)
            diam_losses.append(diam_loss_batch) #added by JGM(add diam_loss_batch to be plotted in tensorboard)
        mean_loss = np.mean(losses)
        mean_diam_loss = np.mean(diam_losses) #added by JGM(add diam_loss_batch to be plotted in tensorboard)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        self.trainer.storage.put_scalar('validation_diam_loss', mean_diam_loss) #add by JGM(add diam_loss_batch to be plotted in tensorboard)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        if os.path.exists('validation_losses.pkl'):
            #load data
            with open('validation_losses.pkl', 'rb') as handle:
                saved_losses = pickle.load(handle)
            #append data
            saved_losses.append(metrics_dict)
            #upgrade data
            with open('validation_losses.pkl', 'wb') as handle:
                pickle.dump(saved_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            saved_losses = []
            saved_losses.append(metrics_dict)
            with open('validation_losses.pkl', 'wb') as handle:
                pickle.dump(saved_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        diam_losses_reduced = metrics_dict['loss_diam'] #add by JGM
        return total_losses_reduced, diam_losses_reduced #modificed by JGM (add diam_losses_reduced)
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
class MyTrainer(DefaultTrainer):
                         
    def build_hooks(self):

        #pdb.set_trace()
        print('BUILD_DETECTION_TEST_MODEL')
        hooks = super().build_hooks()
        #pdb.set_trace()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                mapper
            )
        ))
        return hooks

    @classmethod
    def build_train_loader(cls, cfg):
        print('BUILD_TRAIN_LOADER')
        #pdb.set_trace()
        return build_detection_train_loader(cfg, mapper=mapper)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate detection')
    parser.add_argument('--num_iterations',dest='num_iterations',default=40000,help='maximum number of iterations (not epochs)')
    parser.add_argument('--checkpoint_period',dest='checkpoint_period',default=500,help='save the epoch periodically every X iterations')
    parser.add_argument('--eval_period',dest='eval_period',default=500,help='evaluate the model every X iterations (with the validation set)')
    parser.add_argument('--batch_size',dest='batch_size',default=2)
    parser.add_argument('--learing_rate',dest='learing_rate',default=0.00025)
    parser.add_argument('--experiment_name',dest='experiment_name',default='trial3')
    parser.add_argument('--dataset_path',dest='dataset_path',default='/mnt/gpid07/users/jordi.gene/multitask_RGBD/data/')
    args = parser.parse_args()

    return args
if __name__ == '__main__':
# ------------------------------------------------------HYPERPARAMS TO TEST-----------------------------------------------------
    args = parse_args()
    lr = float(args.learing_rate)
    bs = int(args.batch_size)
    max_iter = int(args.num_iterations)
    checkpoint_period = int(args.checkpoint_period)
    eval_period = int(args.eval_period)
    experiment_name = args.experiment_name
    dataset_path = args.dataset_path
 # ------------------------------------------------------------------------------------------------------------------------------

    print('***EXPERIMENT:***: ',experiment_name)
    print('***BATCH_SIZE: ',bs)
    print('***LR: ',lr)

    print('PREPARING DATA...')
    for d in ["train","val"]:
        DatasetCatalog.register("FujiSfM_" + d, lambda d=d: utils_detection.get_FujiSfM_dicts(dataset_path,d))
        MetadataCatalog.get("FujiSfM_" + d).set(thing_classes=["apple"])
    FujiSfM_metadata = MetadataCatalog.get("FujiSfM_train")
    # Train
    print('TRAIN')
    cfg = get_cfg()
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 1.64]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0,1.42]
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("FujiSfM_train",)
    cfg.DATASETS.TEST =  ("FujiSfM_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    #cfg.MODEL.WEIGHTS = os.path.join(os.getcwd(),'edit_weights.pkl')
    cfg.MODEL.WEIGHTS = os.path.join('edit_weights.pkl')
    #cfg.MODEL.WEIGHTS = os.path.join('/home/usuaris/imatge/mar.ferrer/Fruit_Detectron2_project/recerca/200630-Apple_size_measurement_using_SfM/code/detectron2/output/exp55/model_0005999.pth')
    cfg.SOLVER.IMS_PER_BATCH = bs
    cfg.SOLVER.BASE_LR = lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = max_iter  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.TEST.EVAL_PERIOD = eval_period
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
    cfg.OUTPUT_DIR="./output/"+str(experiment_name)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    if torch.cuda.is_available():
        print('Number of cuda devices = ' + str(torch.cuda.device_count()) + ' ; Current cuda device = ' + torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('Not using cuda devices')
    trainer.train()
    print('FINISHED_TRAINING')






