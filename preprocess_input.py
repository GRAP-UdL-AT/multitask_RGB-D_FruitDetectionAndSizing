from detectron2.utils.logger import setup_logger

setup_logger()
import numpy as np
import os, json, cv2, random
import json
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, GenericMask
import matplotlib.pyplot as plt
import pdb, copy


# Prepare the dataset
def fix_FujiSfM_dicts(root_dir, split):
    json_file = os.path.join(root_dir, 'gt_json', split, "via_region_data_instance.json")

    with open(json_file) as f:
        og_imgs_anns = json.load(f)

    dataset_dicts = []
    imgs_anns = copy.deepcopy(og_imgs_anns)
    for idx, kk in enumerate(og_imgs_anns.keys()):
        v = og_imgs_anns[kk]
        v_new = imgs_anns[kk]
        record = {}
        filename = os.path.join(root_dir, 'images', split, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        annos = v["regions"]
        annos_new = v_new["regions"]
        apple_ids = []
        keys = []
        a = 0
        for key_ in annos.keys():
            anno = annos[key_]
            anno_new = annos_new[key_]
            if 'apple_ID' in anno['region_attributes'].keys():
                appleId = anno['region_attributes']['apple_ID']
                # pdb.set_trace()
                if appleId == '0':
                    annos_new.pop(key_)

                else:
                    if appleId in apple_ids:
                        # pdb.set_trace()
                        id_ap = np.where(np.array(appleId) == np.array(apple_ids))[0][0]
                        rest_x = abs(np.array(annos[keys[id_ap]]['shape_attributes']['all_points_x']) - np.array(
                            annos[key_]['shape_attributes']['all_points_x'][0]))
                        rest_y = abs(np.array(annos[keys[id_ap]]['shape_attributes']['all_points_y']) - np.array(
                            annos[key_]['shape_attributes']['all_points_y'][0]))
                        suma = rest_x + rest_y
                        id_sum = np.argmin(suma)
                        len1 = len(annos[keys[id_ap]]['shape_attributes']['all_points_x'])
                        len2 = len(annos[key_]['shape_attributes']['all_points_x'])
                        new_x = np.zeros((len1 + len2,))
                        new_x[:id_sum] = annos[keys[id_ap]]['shape_attributes']['all_points_x'][:id_sum]
                        new_x[id_sum:id_sum + len2] = annos[key_]['shape_attributes']['all_points_x']
                        new_x[id_sum + len2:] = annos[keys[id_ap]]['shape_attributes']['all_points_x'][id_sum:]
                        annos_new[keys[id_ap]]['shape_attributes']['all_points_x'] = new_x
                        new_y = np.zeros((len1 + len2,))
                        new_y[:id_sum] = annos[keys[id_ap]]['shape_attributes']['all_points_y'][:id_sum]
                        new_y[id_sum:id_sum + len2] = annos[key_]['shape_attributes']['all_points_y']
                        new_y[id_sum + len2:] = annos[keys[id_ap]]['shape_attributes']['all_points_y'][id_sum:]
                        annos_new[keys[id_ap]]['shape_attributes']['all_points_y'] = new_y
                        annos_new.pop(key_)

                        if False:
                            anno_new = annos_new[keys[id_ap]]['shape_attributes']
                            px = anno_new["all_points_x"]
                            py = anno_new["all_points_y"]
                            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                            poly = [p for x in poly for p in x]
                            Gmask = GenericMask(poly, height, width)
                            gt_mask_sing = Gmask.polygons_to_mask([poly])
                            pdb.set_trace()
                            plt.imsave(v["filename"] + key_ + '.png', gt_mask_sing)
                            # assert not anno["region_attributes"]
                    else:

                        apple_ids.append(appleId)
                        keys.append(key_)

        if len(annos_new.keys()) < 1:
            imgs_anns.pop(kk)
    return imgs_anns