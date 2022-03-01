#from typing import final
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
import json
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, GenericMask
import matplotlib.pyplot as plt
import pdb
import preprocess_input

#Prepare the dataset
def get_FujiSfM_dicts(root_dir,split):
  
    depth_path = root_dir+'depthCropNpy'
    diameters_path = root_dir+'GT_diameter.txt'
    json_file_i = os.path.join(root_dir+'gt_json/'+split, "via_region_data_instance.json")
    json_file_a = os.path.join(root_dir+'gt_json/'+split, "via_region_data_amodal.json")
  
    imgs_anns_inst = preprocess_input.fix_FujiSfM_dicts(root_dir,split)
    with open(json_file_a) as f:
        imgs_anns_amod = json.load(f)
        
    with open(diameters_path) as f:
        lines = f.readlines()
        
    list_2020 = []
    list_2018 = []
    for line in lines:
        if '2020' in line:
            list_2020.append(line)
        else:
            list_2018.append(line)

    dataset_dicts = []
    wrong_ids = []
    for idx, v in enumerate(imgs_anns_inst.values()):
        record = {}
        filename = os.path.join(root_dir+'images/'+split, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        name = filename.split('/')[-1]
        name_amodal = name+str(v['size'])
        name = name.split('.')[0]
        
        # Load amodal info
        v_a = imgs_anns_amod[name_amodal]

        record["file_name"] = filename
        record["depth_file"] = os.path.join(depth_path,name+'.npy')
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        annos = v["regions"]
        annos_a = v_a["regions"]
        objs = []
        a = 0
        for key_ in annos.keys():
            anno = annos[key_]
            if 'apple_ID' in anno['region_attributes'].keys():

                appleId = anno['region_attributes']['apple_ID']
                for key_a in annos_a.keys():
                    appleId_amod = annos_a[key_a]['region_attributes']['apple_ID']

                    if appleId == appleId_amod:
                       
                        anno_a = annos_a[key_a]
                        sem = 1
                        apple_ID = anno['region_attributes']['apple_ID']
                        num = name.split('_')[2]
                        if num[0] == '6' or num[0] == '7':
                            
                            # This means 2020. Format: 2020_01_322.txt,63
                            s = '2020_01_'+apple_ID.zfill(3)+'.txt'
                            corr_line = [ l for l in list_2020 if s in l]
                            
                            if corr_line == []:
                                print('2020',apple_ID)
                                print(name)
                                wrong_ids.append('2020_'+apple_ID)
                                corr_line = ['blabla,50/n']
                            
                            n = corr_line[0].split(',')
                            n = n[-1].split('/n')[0]
                            diameter = float(n)/100 
                                
                        elif num[0] == '2' or num[0]=='3':
                            # This means 2018
                            s = '2018_01_'+apple_ID.zfill(3)+'.txt'
                            corr_line = [ l for l in list_2018 if s in l]
                            
                            if corr_line == []:
                                print('2018',apple_ID)
                                print(name)
                                wrong_ids.append('2018_'+apple_ID)
                                corr_line = ['blabla,50/n']
                            n = corr_line[0].split(',')
                            n = n[-1].split('/n')[0]
                            diameter = float(n)/100   
                                   
                            
                            
                        anno = anno["shape_attributes"]
                        anno_a = anno_a['shape_attributes']
                        px = anno["all_points_x"]
                        py = anno["all_points_y"]
                        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                        poly = [p for x in poly for p in x]
                        # Amodal
                        px_a = anno_a["all_points_x"]
                        py_a = anno_a["all_points_y"]
                        poly_a = [(x + 0.5, y + 0.5) for x, y in zip(px_a, py_a)]
                        poly_a = [p for x in poly_a for p in x]
                                        
                        if len(poly)>4 and len(poly_a)>4:
                            
                            obj = {
                                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                                "bbox_amodal": [np.min(px_a), np.min(py_a), np.max(px_a), np.max(py_a)],
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "segmentation": [poly],
                                "amodal_segmentation":[poly_a],
                                "diameter":[diameter],
                                "category_id": 0,
                                "appleId":apple_ID
                            }
                            objs.append(obj)
          
        record["annotations"] = objs
        dataset_dicts.append(record)
            

    return dataset_dicts

def remove_margin_detections(dict_pred,idx_match):
 
    horitz  = int(round(192/2))
    vert = int(round(213/2))
    i_ = 0
    dict_pred['remove'] = np.ones(len(dict_pred['pred'])) == 4
    filt_pred = {}
    filt_pred['pred'] = []
    filt_pred['box'] = []
    
    for i in range(len(dict_pred['pred'])):
        pred_mask = dict_pred['pred'][i]
        crop_pred =pred_mask[vert:np.shape(pred_mask)[0]-vert,horitz:np.shape(pred_mask)[1]-horitz]
        total_area = sum(sum(crop_pred))
        crop_pred = cv2.resize((crop_pred*1).astype('float32'), (np.shape(pred_mask)[1],np.shape(pred_mask)[0]), interpolation=cv2.INTER_AREA)
        dict_pred['pred'][i] = crop_pred

        if total_area<70:
            dict_pred['remove'][i] = True
    
    for i in range(len(dict_pred['pred'])):
      
        if not dict_pred['remove'][i]:
            filt_pred['pred'].append(dict_pred['pred'][i])
            filt_pred['box'].append(dict_pred['box'][i])
        else:
            if idx_match is not None:
                del idx_match[i_]
                i_ = i_-1
        i_+= 1    
   
    return filt_pred,dict_pred,idx_match

def remove_margins_gt(dataset_dicts,iter_,im,og_num_gt):
  
    horitz  = int(round(192/2))
    vert = int(round(213/2))
    dict_gt = {}
    dict_gt['remove'] = np.ones(len(dataset_dicts[0][iter_]['annotations'])) == 4
    dict_gt['mask'] = []
    dict_gt['box'] = []
    dict_gt['amodal_mask'] = []
    og_num_gt += len(dataset_dicts[0][iter_]['annotations'])

    for jj, gt_mask in enumerate(dataset_dicts[0][iter_]['annotations']):
       
        gt_poly = gt_mask['segmentation']
        gt_box = gt_mask['bbox']
        gt_amodal_mask = gt_mask['amodal_segmentation']
        Gmask = GenericMask(gt_poly, im.shape[0], im.shape[1])
        gt_mask_sing = Gmask.polygons_to_mask(gt_poly)
        crop_gt = gt_mask_sing[vert:np.shape(gt_mask_sing)[0]-vert,horitz:np.shape(gt_mask_sing)[1]-horitz]
        total_area = sum(sum(crop_gt))
        crop_gt = cv2.resize(crop_gt, (np.shape(im)[1],np.shape(im)[0]), interpolation=cv2.INTER_AREA)

        # amodal mask
        Gmask_a = GenericMask(gt_amodal_mask, im.shape[0], im.shape[1])
        gt_mask_sing_a = Gmask.polygons_to_mask(gt_amodal_mask)
        crop_gt_a = gt_mask_sing_a[vert:np.shape(gt_mask_sing)[0]-vert,horitz:np.shape(gt_mask_sing)[1]-horitz]
        total_area_a = sum(sum(crop_gt_a))
        crop_gt_a = cv2.resize(crop_gt_a, (np.shape(im)[1],np.shape(im)[0]), interpolation=cv2.INTER_AREA)
        # Remove the mask if the area is too small
        if total_area<70:
            dict_gt['remove'][jj] = True
        
        dict_gt['mask'].append(crop_gt)
        dict_gt['box'].append(np.array(gt_box))
        dict_gt['amodal_mask'].append(crop_gt_a)


    gt_keep = []
    gt_box_keep = []
    gt_amodal_keep = []
    for i in range(len(dict_gt['mask'])):
      
        if not dict_gt['remove'][i]:
            gt_keep.append(dict_gt['mask'][i])
            gt_box_keep.append(dict_gt['box'][i])
            gt_amodal_keep.append(dict_gt['amodal_mask'][i])

    return gt_keep, gt_box_keep, og_num_gt


def match_amodal_instance(pred_amodal,pred_instance,pred_box,diameter,scores):
  
    new_instance = []
    new_amodal = []
    new_diameter = []
    new_scores = []
    occ_prop = []
    idx_match = []
    new_box = []
    if len(pred_amodal) < len(pred_instance):
        shortest = len(pred_amodal)
    elif len(pred_amodal) > len(pred_instance):
        shortest =  len(pred_instance)
    elif len(pred_amodal) == len(pred_instance):
        shortest = len(pred_amodal)

    for i,ma in enumerate(pred_amodal):
        max_occ = 0
        for j,mi in enumerate(pred_instance):
            occ = mi*1+ma*1
            occ = occ == 2
            sum_amod = sum(sum(ma*1))
            occlusion_prop = sum(sum(occ))/sum_amod

            if occlusion_prop>max_occ:
                max_occ = occlusion_prop
                max_i = i
                max_j = j
        
        if max_occ > 0:
            new_amodal.append(pred_amodal[max_i])
            new_instance.append(pred_instance[max_j])
            new_diameter.append(diameter[max_j])
            new_scores.append(scores[max_j])
            occ_prop.append(max_occ)
            idx_match.append([max_i,max_j])
            new_box.append(pred_box[max_j])
      
    return new_instance, new_amodal,new_box, occ_prop,idx_match,new_diameter,new_scores



def match_mask(pred_inst,pred_box,pred_diameter,dict_pred,dataset_dicts,im, gt_num_masks,newIm,iter_,pred_amodal,diameter,scores):
    used_ids_gt = []
    all_sum = []
    gts = []
    gts_box = []
    gts_amod_mask= []
    g_diam = []
    g_id = []
    amod = []
    final_preds = []
    final_box = []
    final_amodals = []
    final_scores = []
    final_diameter = []
    num_gt = len(dataset_dicts[0][iter_]['annotations'])
   
    
    for i in range(len(pred_inst)):
        pred_mask = pred_inst[i]
        max_sum = 0
        if num_gt>=1:
            for j, gt_poly in enumerate(dataset_dicts[0][iter_]['annotations']):
                gt_poly = gt_poly['segmentation']
                Gmask = GenericMask(gt_poly, im.shape[0], im.shape[1])
                gt_mask = Gmask.polygons_to_mask(gt_poly)
                mult = gt_mask*pred_mask
                suma = sum(sum(mult))
                if suma >= max_sum:
                    idx_m = j
                    idx_p = i
                    max_sum = suma
            final_preds.append(pred_inst[idx_p])
            final_box.append(pred_box[idx_p])
            final_diameter.append(pred_diameter[idx_p])
            final_scores.append(scores[idx_p])
            gt_diam = dataset_dicts[0][iter_]['annotations'][idx_m]['diameter']
            gt_id = dataset_dicts[0][iter_]['annotations'][idx_m]['appleId']
            gt_poly_f = dataset_dicts[0][iter_]['annotations'][idx_m]['segmentation']
            Gmask = GenericMask(gt_poly_f, im.shape[0], im.shape[1])
            gt_mask_sing = Gmask.polygons_to_mask(gt_poly_f)
            gt_box_sing = dataset_dicts[0][iter_]['annotations'][idx_m]['bbox']
           
            if idx_m in used_ids_gt:
                a = np.where(np.array(used_ids_gt) == idx_m)[0][0]
                if all_sum[a]<max_sum:
                    gts[a] = None
                    used_ids_gt[a] = None
                    g_diam[a] = None
                    g_id[a]=None
                    gts_box[a] = None

                else:
                    idx_m = None
                    gt_mask_sing = None
                    gt_box_sing = None
                    gt_diam = None
                    gt_id = None


            used_ids_gt.append(idx_m)
            all_sum.append(max_sum)
            gts.append(gt_mask_sing)
            gts_box.append(gt_box_sing)
            g_diam.append(gt_diam)
            g_id.append(gt_id)
        else:
            gts.append(None)
            gts_box.append(None)
            g_diam.append(None)
            g_id.append(None)

    return gts,gts_box,num_gt,g_diam,final_preds,final_box,None,final_diameter,final_scores

def match_mask_removing_margins(filt_pred,dict_pred,dataset_dicts,gt_keep, gt_box_keep,im, gt_num_masks,newIm,iter_,pred_amodal,diameter,scores):
    #returns pred_mask with higher intersection with the gt (performs NMS)
    
    used_ids_gt = []
    all_sum = []
    gts = []
    gts_box = []
    gts_amod_mask= []
    g_diam = []
    g_id = []
    amod = []
    final_preds = []
    final_box = []
    final_amodals = []
    final_scores = []
    final_diameter = []
    num_gt = len(gt_keep)
    if len(dict_pred.keys()) == 2:
        dict_pred['remove'] = np.ones(len(dict_pred['pred'])) == 4
        if pred_amodal is not None:
            for i in range(len(dict_pred['remove'])):
                if not dict_pred['remove'][i]:
                    final_amodals.append(pred_amodal[i])
                    final_diameter.append(diameter[i])
                    final_scores.append(scores[i])
            
    elif pred_amodal is not None:
        if len(pred_amodal) != len(dict_pred['remove']):
            print('ERROR in match masks, pred_amodal and dict_pred should have the same length')
        
        for i in range(len(dict_pred['remove'])):
            if not dict_pred['remove'][i]:
                final_amodals.append(pred_amodal[i])
                final_diameter.append(diameter[i])
                final_scores.append(scores[i])
    elif pred_amodal is None: 
        for i in range(len(dict_pred['remove'])):
            if not dict_pred['remove'][i]:
                final_diameter.append(diameter[i])
                final_scores.append(scores[i])


    for i in range(len(filt_pred['pred'])):
        
        pred_mask = filt_pred['pred'][i]
        final_preds.append(pred_mask)
        final_box.append(filt_pred['box'][i])
        max_sum = 0
        if num_gt>=1:
            for j, gt_mask in enumerate(gt_keep):
                mult = gt_mask*pred_mask
                suma = sum(sum(mult))
                if suma >= max_sum:
                    idx_m = j
                    idx_p = i
                    max_sum = suma
            
            gt_diam = dataset_dicts[0][iter_]['annotations'][idx_m]['diameter']
            gt_id = dataset_dicts[0][iter_]['annotations'][idx_m]['appleId']
            gt_mask_sing = gt_keep[idx_m]
            gt_box_sing = gt_box_keep[idx_m]
            if idx_m in used_ids_gt:
                a = np.where(np.array(used_ids_gt) == idx_m)[0][0]
                if all_sum[a]<max_sum:
                    gts[a] = None
                    used_ids_gt[a] = None
                    g_diam[a] = None
                    g_id[a]=None
                    gts_box[a] = None

                else:
                    idx_m = None
                    gt_mask_sing = None
                    gt_box_sing = None
                    gt_diam = None
                    gt_id = None
                   


            used_ids_gt.append(idx_m)
            all_sum.append(max_sum)
            gts.append(gt_mask_sing)
            gts_box.append(gt_box_sing)
            g_diam.append(gt_diam)
            g_id.append(gt_id)
           
        else:
            gts.append(None)
            gts_box.append(None)
            g_diam.append(None)
            g_id.append(None)

    
    return gts,gts_box,num_gt,g_diam,final_preds,final_box,None,final_diameter,final_scores


def plot_masks(mask_plot,gt_mask,im):
    for i in range(len(gt_mask)):
        if gt_mask[i] is not None:
            # mask_plot[:, :, 0] = gt_mask_sing*255 + mask_plot[:, :, 0]
            # mask_plot[:, :, 1] = gt_mask_sing*255 + mask_plot[:, :, 1]
            mask_plot[:, :, 2] = gt_mask[i]*255 + mask_plot[:, :, 2]

    return mask_plot

#https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/pascal_voc_evaluation.py
def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """

    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))


    # intersection over union
    intersection = np.logical_and(masks1, masks2)
    union = np.logical_or(masks1, masks2)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


def iou_bbox(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def cumpute_occlusion_dual(idx_match, all_errors, diam_err_per_im,j):
    
    curr_i =  idx_match[j][0]
    #print('curri',curr_i)
    curr_j = idx_match[j][1]
    iid =  np.where(curr_i==np.array(idx_match).transpose()[0])
    jjd =  np.where(curr_j==np.array(idx_match).transpose()[1])
    if len(iid) > 1 or len(jjd)>1:
        mindiam_i = np.argmin(diam_err_per_im[iid])
        mindiam_j = np.argmin(diam_err_per_im[jjd])
    else:
        mindiam_i = mindiam_j = iid[0][0]
    
    if mindiam_i == mindiam_j:
        error_diam = diam_err_per_im[mindiam_i]
    else:
        if diam_err_per_im[mindiam_i]<diam_err_per_im[mindiam_j]:
            error_diam = diam_err_per_im[mindiam_i]
        else:
            error_diam = diam_err_per_im[mindiam_j]
    
    return error_diam

def gt_occ(gt_inst,gt_amod):
    occ_ = gt_amod*1+gt_inst*1
    occ_ = occ_ == 2
    sum_amod = sum(sum(gt_amod))
    occlusion_prop = sum(sum(occ_))/sum_amod
    return occlusion_prop


def extract_polys(mask):
    mm,contours, h = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    poly = {}
    if len(contours) > 1:
        base = np.load('base_cont.npy', allow_pickle=True)
        for j, cont in enumerate(contours):
            if j > 0:
                og = np.concatenate((og, cont), axis=0)
            else:
                og = cont
        base[0] = og
        contours= base
    for i,cont in enumerate(contours):
        poly[i] = {}
        poly[i]['all_points_x'] = cont[:, 0, 0]
        poly[i]['all_points_y'] = cont[:, 0, 1]
    
    return poly
