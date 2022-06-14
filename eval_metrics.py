import numpy as np
import cv2
import os
import utils_diameter
from tqdm import trange



def prec_rec_f1_ap_MAE(predictor, dataset_dicts, path, confidence_scores,split,iou_thr,save_metrics,output_dir,year='all'):
    save_path = output_dir+'/results_txt/'+split+'/'
    if save_metrics:
        if not os.path.exists(output_dir+'/results_txt/'):
            os.mkdir(output_dir+'/results_txt/')

    names = []
    tp_all = np.array([])
    fp_all = np.array([])
    tp_count = np.zeros(len(confidence_scores))
    fp_count = np.zeros(len(confidence_scores))
    gt_num_masks = 0
    diam_err = []
    init_diam_err = np.ones(len(confidence_scores))
    for i in trange(len(dataset_dicts[0])):
       
        dict_info_txt = {}
        iter_ = i
        newIm = True
        file_name = dataset_dicts[0][i]['file_name']
        depth_map = np.load(dataset_dicts[0][i]["depth_file"])
        
        im_name = file_name.split('/')[-1]

        if year == '2018': #by JGM
            if int(im_name[4]) > 4:
                continue

        if year == '2020': #by JGM
            if int(im_name[4]) < 4:
                continue
        print(im_name) #by JGM
        if True:
            names.append(im_name)
            file_name = os.path.join(path, im_name)
            # gt_annots = dataset_dicts[0][i]['annotations']
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

            dict_pred = {'pred': pred_instance, 'box':pred_box, 'isUsed': np.ones(len(pred_instance)) * False}
            gts,gts_box, gts_amod_mask,num_gt, gt_diam,gt_id,final_preds,final_box,final_amods,final_diameter,final_scores = utils_diameter.match_mask(pred_instance, pred_amodal,pred_box,diameter,scores,dict_pred,dataset_dicts,im, gt_num_masks,newIm,iter_)
            gt_num_masks += num_gt

            # tp_temp = np.zeros((len(pred_instance),len(confidence_scores)))
            # fp_temp = np.zeros((len(pred_instance),len(confidence_scores)))
            # occ = []

            for j, pred_mask in enumerate(final_preds):
                poly_txt = utils_diameter.extract_polys(pred_mask)
                dict_info_txt['all_points_x'] = str(poly_txt[0]['all_points_x']).replace('\n','')
                dict_info_txt['all_points_y'] = str(poly_txt[0]['all_points_y']).replace('\n','')
                dict_info_txt['conf'] = str(final_scores[j])
                dict_info_txt['diam'] = str(final_diameter[j][0])
                dict_info_txt['gt_diam'] = '0'
                dict_info_txt['id'] = '0'
                dict_info_txt['occ']='0'
                dict_info_txt['gt_occ'] = '0'
                dict_info_txt['iou'] = '0'
                if gt_id[j] != None:
                    amod_crop = gts_amod_mask[j]
                    iou = utils_diameter.iou_bbox(gts_box[j], final_box[j])
                    if iou >= iou_thr:
                        occ_ = amod_crop * 1 + pred_mask * 1
                        occ_ = occ_ == 2
                        sum_amod = sum(sum(amod_crop))
                        occlusion_prop = sum(sum(occ_)) / sum_amod
                        occ_gt = utils_diameter.gt_occ(gts[j], gts_amod_mask[j])
                        dict_info_txt['gt_diam'] = str(gt_diam[j][0])
                        dict_info_txt['id'] = gt_id[j]
                        dict_info_txt['occ']= str(1-occlusion_prop)
                        dict_info_txt['gt_occ'] = str(1-occ_gt)
                        dict_info_txt['iou'] = str(iou)
                        error_diam = abs(gt_diam[j][0] - final_diameter[j][0])
                        tp_all = np.concatenate((tp_all,[1]))
                        fp_all = np.concatenate((fp_all,[0]))
                    else:
                        tp_all = np.concatenate((tp_all,[0]))
                        fp_all = np.concatenate((fp_all,[1]))
                else:
                    iou=0
                    tp_all = np.concatenate((tp_all,[0]))
                    fp_all = np.concatenate((fp_all,[1]))

                if save_metrics:
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    with open(save_path + im_name.replace('.png', '.txt'), 'a+') as f:
                        f.write("\n")
                        f.write(
                            dict_info_txt['id'] + '|' + dict_info_txt['conf'] + '|' + dict_info_txt['diam'] + '|' +
                            dict_info_txt['gt_diam'] + '|' + dict_info_txt['occ'] + '|' + dict_info_txt[
                                'gt_occ'] + '|' + dict_info_txt['all_points_x'] + '|' + dict_info_txt[
                                'all_points_y']+ '|' + dict_info_txt['iou'])

                for k, s in enumerate(confidence_scores):
                    if final_scores[j] >= s:
                        if iou >= iou_thr:
                            if init_diam_err[k]:
                                diam_err.append([])
                                init_diam_err[k] = False
                            # tp_temp[j][k] = 1.0
                            tp_count[k] += 1
                            diam_err[k].append(error_diam)
                        else:
                            # fp_temp[j][k] = 1.0
                            fp_count[k] += 1


        # tp_all = np.concatenate((tp_all, tp_temp), axis=0)
        # fp_all = np.concatenate((fp_all, fp_temp), axis=0)
    fp = np.cumsum(fp_all)
    tp = np.cumsum(tp_all)
    rec = tp / float(gt_num_masks)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = utils_diameter.voc_ap(rec, prec, True)
    P = np.zeros(len(confidence_scores))
    R = np.zeros(len(confidence_scores))
    F1 = np.zeros(len(confidence_scores))
    AP = np.zeros(len(confidence_scores))
    MAE = np.zeros(len(confidence_scores))
    for k, s in enumerate(confidence_scores):
        # fp_s = np.delete(fp_all[:,k], tp_all[:,k]==fp_all[:,k])
        # tp_s = np.delete(tp_all[:,k], tp_all[:,k]==fp_all[:,k])
        # fp = np.cumsum(fp_s)
        # tp = np.cumsum(tp_s)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        AP[k] = ap
        P[k] = tp_count[k] / (tp_count[k] + fp_count[k])
        R[k] = tp_count[k] / (gt_num_masks)
        F1[k] = (2 * P[k] * R[k]) / (P[k] + R[k])
        MAE[k] = np.mean(diam_err[k])
    print('Confidence socres:', confidence_scores)
    print('PRECISION:', P)
    print('RECALL:', R)
    print('F1:', F1)
    print('AP:', AP)
    print('MAE:', MAE)

    results_lines = [ 'Conf,' + ",".join(np.char.mod('%f',confidence_scores)),
                      'P   ,' + ",".join(np.char.mod('%f',P)),
                      'R   ,' + ",".join(np.char.mod('%f',R)),
                      'F1  ,' + ",".join(np.char.mod('%f',F1)),
                      'AP  ,' + ",".join(np.char.mod('%f',AP)),
                      'MAE ,' + ",".join(np.char.mod('%f',MAE))]
    with open(output_dir+"/"+save_path.split('/')[3]+"_P_R_F1_AP_MAE_results.csv", 'w') as f:
        f.write('\n'.join(results_lines))



    return P, R, F1, AP, MAE

