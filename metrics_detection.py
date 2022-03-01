import numpy as np
import cv2
import os
import utils_detection
from detectron2.utils.visualizer import Visualizer, GenericMask
from detectron2.utils.visualizer import ColorMode
from tqdm import trange
import matplotlib.pyplot as plt
import pdb


def prec_rec_f1_ap(predictor, dataset_dicts, path, score_thr,FujiSfM_metadata,iou_thr):

    names = []
    tp = np.array([[]])
    fp = np.array([[]])
    tp_count = 0
    fp_count = 0
    fp_count_none = 0
    gt_num_masks = 0
    final_num_masks = 0
    og_num_gt = 0
    total_error_diam = 0
    occlusion = []
    all_errors = []
    for i in trange(len(dataset_dicts[0])):

        iter_ = i
        newIm = True
        file_name = dataset_dicts[0][i]['file_name']
        depth_map = np.load(dataset_dicts[0][i]["depth_file"])
        
        im_name = file_name.split('/')[-1]
        if im_name[:5] == '_MG_2' or im_name[:5] == '_MG_3' or im_name[:5] == '_MG_6' or im_name[:5] == '_MG_7':

            names.append(im_name)
            file_name = os.path.join(path, im_name)
            gt_annots = dataset_dicts[0][i]['annotations']
            im = cv2.imread(file_name)
            
            with open('current_images_inference.txt', 'w') as f:
                f.write("%s\n" % im_name)   
        
            try:
                depth_map = cv2.resize(depth_map, (np.shape(im[:,:,0])[1],np.shape(im[:,:,0])[0]), interpolation=cv2.INTER_AREA)
        
            except:
                raise Exception("COULD NOT RESIZE"+dataset_dicts['depth_file'])
            
            image = np.zeros((np.shape(im)[0],np.shape(im)[1],4))
            image[:,:,0:3] = im
            image[:,:,3] = depth_map
            mask_plot = np.zeros(im.shape)
            pred_plot = np.zeros(im.shape)

            outputs = predictor(image)

           
            predictions = outputs["instances"].to("cpu")
        
            scores = predictions.scores.tolist()
            #print('OG_SCORES:',scores)
            diameter = predictions.pred_diameter.tolist()
            pred_instance = np.asarray(predictions.pred_masks)#np.asarray(predictions.pred_amodal)
            pred_box = predictions.pred_boxes.tensor.numpy()
    
            dict_pred = {'pred': pred_instance, 'box':pred_box, 'isUsed': np.ones(len(pred_instance)) * False}
           
            gts,gts_box, num_gt, gt_diam,final_preds,final_box,final_amods,final_diameter,final_scores = utils_detection.match_mask(pred_instance,pred_box,diameter,dict_pred,dataset_dicts,im, gt_num_masks,newIm,iter_,None,diameter,scores)
            #print('FINAL_SCORES:',final_scores)
            mask_plot = utils_detection.plot_masks(mask_plot, gts, im)
            gt_num_masks += num_gt
            diam_err_per_im = []
            tp_bool=[]
            occ = []
            tp_temp = np.zeros((1, len(final_preds)))
            fp_temp = np.zeros((1, len(final_preds)))
            for j, pred_mask in enumerate(final_preds):
                
                if final_scores[j] >= score_thr:
                    if gts[j] is not None:
                        iou = utils_detection.iou_bbox(gts_box[j], final_box[j])
                        if iou >= iou_thr:
                            # If there is a match, we compute the precision and recall
                            tp_temp[0][j] = 1.0
                            pred_plot[:, :, 1] = pred_mask * 255 + pred_plot[:, :, 1]
                            tp_count += 1
                            tp_bool.append(True)
                        else:
                            tp_bool.append(False)
                            fp_temp[0][j] = 1.0
                            fp_count += 1
                    else:
                        tp_bool.append(False)
                        fp_temp[0][j] = 1.0
                        fp_count_none += 1
                else:
                    tp_bool.append(False)
                newIm = False
            tp = np.concatenate((tp, tp_temp), axis=1)
            fp = np.concatenate((fp, fp_temp), axis=1)
   
   
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(gt_num_masks)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = utils_detection.voc_ap(rec, prec, True)
    fp_count = fp_count+fp_count_none
    P = tp_count / (tp_count + fp_count)
    R = tp_count / (gt_num_masks)
    F1 = (2 * P * R) / (P + R)
    mean_error_diam = np.mean(all_errors)
    std_diam = np.std(all_errors)
    print('PRECISION:', P)
    print('RECALL:', R)
    print('F1:', F1)

    return P, R, F1, ap
