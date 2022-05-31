import numpy as np
import cv2
import os
import utils_diameter
from detectron2.utils.visualizer import Visualizer, GenericMask
from detectron2.utils.visualizer import ColorMode
from tqdm import trange
import matplotlib.pyplot as plt
import pdb,random


def prec_rec_f1_ap(predictor, dataset_dicts, path, score_thr,FujiSfM_metadata,split,iou_thr,save_metrics,output_dir):
    save_path = output_dir+'/results_txt/'+split+'/'
    if save_metrics:
        if not os.path.exists(output_dir+'/results_txt/'):
            os.mkdir(output_dir+'/results_txt/')

    names = []
    tp = np.array([[]])
    fp = np.array([[]])
    tp_count = 0
    fp_count = 0
    fp_count_none = 0
    gt_num_masks = 0
    final_num_masks = 0
    og_num_gt = 0
    occlusion = []
    all_occ = []
    all_errors = []
    for i in trange(len(dataset_dicts[0])):
       
        dict_info_txt = {}
        iter_ = i
        newIm = True
        file_name = dataset_dicts[0][i]['file_name']
        depth_map = np.load(dataset_dicts[0][i]["depth_file"])
        
        im_name = file_name.split('/')[-1]

        if True:
            names.append(im_name)
            file_name = os.path.join(path, im_name)
            gt_annots = dataset_dicts[0][i]['annotations']
            im = cv2.imread(file_name)
            
            with open(os.path.join(output_dir,'current_images_inference.txt'), 'w') as f:
                f.write("%s\n" % im_name)   
            
            try:
                depth_map = cv2.resize(depth_map, (np.shape(im[:,:,0])[1],np.shape(im[:,:,0])[0]), interpolation=cv2.INTER_AREA)
        
            except:
                raise Exception("COULD NOT RESIZE"+dataset_dicts['depth_file'])
            
            image = np.zeros((np.shape(im)[0],np.shape(im)[1],4))
            image[:,:,0:3] = im
            image[:,:,3] = depth_map

            outputs = predictor(image)

            predictions = outputs["instances"].to("cpu")
            scores = predictions.scores.tolist()
            diameter = predictions.pred_diameter.tolist()
            pred_instance = np.asarray(predictions.pred_masks)
            pred_amodal = pred_instance
            pred_box = predictions.pred_boxes.tensor.numpy()
           
            ## Uncomment if using separate branches
            pred_instance, pred_amodal,pred_box,occ,idx_match,diameter,scores = utils_diameter.match_amodal_instance(pred_amodal,pred_instance,pred_box,diameter,scores)

            tp_temp = np.zeros((1, len(pred_instance)))
            fp_temp = np.zeros((1, len(pred_instance)))
            dict_pred = {'pred': pred_instance, 'box':pred_box, 'isUsed': np.ones(len(pred_instance)) * False}
            gts,gts_box, gts_amod_mask,num_gt, gt_diam,gt_id,final_preds,final_box,final_amods,final_diameter,final_scores = utils_diameter.match_mask(pred_instance, pred_amodal,pred_box,diameter,scores,dict_pred,dataset_dicts,im, gt_num_masks,newIm,iter_)
            gt_num_masks += num_gt
            tp_bool=[]
            occ = []
            
            for j, pred_mask in enumerate(final_preds):
                poly_txt = utils_diameter.extract_polys(pred_mask)
                
                dict_info_txt['all_points_x'] = str(poly_txt[0]['all_points_x']).replace('\n','')
                dict_info_txt['all_points_y'] = str(poly_txt[0]['all_points_y']).replace('\n','')
                dict_info_txt['conf'] = str(final_scores[j])
                dict_info_txt['gt_diam'] = '0'
                dict_info_txt['diam'] = '0'
                dict_info_txt['id'] = '0'
                dict_info_txt['occ']='0'
                dict_info_txt['gt_occ'] = '0'

                if final_scores[j] >= score_thr:
                    dict_info_txt['conf'] = str(final_scores[j])
                    if gts[j] is not None:
                        iou = utils_diameter.iou_bbox(gts_box[j], final_box[j])
                        if iou >= iou_thr:
                            # If there is a match, we compute the precision and recall
                            tp_temp[0][j] = 1.0
                            tp_count += 1
                            tp_bool.append(True)

                            # COMPUTE DIAMETER ERROR
                            
                            error_diam = abs(gt_diam[j][0] - final_diameter[j][0])
                            all_errors.append(error_diam)
                         
                            #Using amodal ground truth 
                            amod_crop = gts_amod_mask[j]
                            occ_ = amod_crop*1+pred_mask*1
                            occ_ = occ_ == 2
                            sum_amod = sum(sum(amod_crop))
                            occlusion_prop = sum(sum(occ_))/sum_amod
                            occ.append(occlusion_prop)
                            all_occ.append(occ)
                            
                            # calculate the gt occlusion
                            occ_gt = utils_diameter.gt_occ(gts[j],gts_amod_mask[j])

                            # update the dict
                            dict_info_txt['diam'] = str(final_diameter[j][0])
                            dict_info_txt['gt_diam'] = str(gt_diam[j][0])
                            dict_info_txt['id'] = gt_id[j]
                            dict_info_txt['occ']= str(1-occlusion_prop)
                            dict_info_txt['gt_occ'] = str(1-occ_gt)
                            

                        else:
                            tp_bool.append(False)

                            fp_temp[0][j] = 1.0
                            fp_count += 1
                            dict_info_txt['gt_diam'] = '0'
                            dict_info_txt['diam'] = '0'
                            dict_info_txt['id'] = '0'
                            dict_info_txt['occ']='0'
                            dict_info_txt['gt_occ'] = '0'
                    

                    else:
                        tp_bool.append(False)
                        fp_temp[0][j] = 1.0
                        fp_count_none += 1
                        dict_info_txt['diam'] = '0'
                        dict_info_txt['gt_diam'] = '0'
                        dict_info_txt['id'] = '0'
                        dict_info_txt['occ']='0'
                        dict_info_txt['gt_occ'] = '0'

                    if save_metrics:
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        with open(save_path+im_name.replace('.png','.txt'),'a+') as f:
                            f.write("\n")
                            f.write(dict_info_txt['id']+'|'+dict_info_txt['conf']+'|'+dict_info_txt['diam']+'|'+dict_info_txt['gt_diam']+'|'+dict_info_txt['occ']+'|'+dict_info_txt['gt_occ']+'|'+ dict_info_txt['all_points_x']+'|'+dict_info_txt['all_points_y'])

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
    ap = utils_diameter.voc_ap(rec, prec, True)
    fp_count = fp_count+fp_count_none
    P = tp_count / (tp_count + fp_count)
    R = tp_count / (gt_num_masks)
    F1 = (2 * P * R) / (P + R)
    MAE = np.mean(all_errors)
    print('PRECISION:', P)
    print('RECALL:', R)
    print('F1:', F1)
    print('MAE:',MAE)

    return P, R, F1, ap, MAE


