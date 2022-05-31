# Multitask Deep Neural Network for Fruit Detection and Regresion of Fruit Diameters in RGB-D images (based on Detectron2)

## Introduction
This project is an extension of the MaskRCNN architecture that allows to compute the diameter of fruits along with performing instance segmentation. The baseline for this project has been the detectron2 implementation of the MaskRCNN (https://github.com/facebookresearch/detectron2).

## Preparation 


The python version used in this project is Python 3.6.13. 
The first step is to clone the code into your own directory with git clone. 

Then, download the folder *data_fse* that contains the dataset along with the annotations and other necessary files.
```
http://www.grap.udl.cat/en/publications/PApple_RGB-D-Size.html
```
The structure of your files should be the following:

--detectron2  
--training_maskrcnn.py  
--inference.py  
--metrics_detection.py  
--utils_detection.py  
--metrics_diamteter.py  
--edit_weights.pkl  
--utils_diameter.py  
--data_fse  
----images   
----gt_json  
----depthCropNpy  
----GT_diameters.txt  




### Prerequisites
```
pip install -r requirements.txt
```

## Training

This section presents how to train a model that is able to detect, segment and measure apples in the dataset. If the files are organised in the indicated way, *training_maskrcnn.py* is ready to be used directly. 

```
python train_maskrcnn.py
```
The default parameters are the following:
- num_iterations (maximum number of iterations, not epochs) --> 10000
- checkpoint_period (the model will be stored periodically every X iterations) --> 1000
- eval_period (the model will be evaluated with the validation set every X iterations) --> 1000
- batch_size --> 2
- learning_rate --> 0.00025
- experiment_name (name you want to assing to the experiment) --> "trial"
- dataset path (path to your data folder) --> "data_fse/"

Example: modifying the default learning rate and the batch size:

```
python train_maskrcnn.py --learning_rate 0.001 --batch_size 4
```

Once the training is complete, the experiment folder containing the epochs will be stored inside an output folder:
--output
----"name_experiment"
------epoch_0.pth
------epoch_1.pth
------...

## Inference

This model performs 2 taks in one: 1) detection, 2)diameter estimation. This code tests the 2 tasks separately. 

### Inference on detection

It will reveal the Precision, Recall, F1-score scores in the fruit detection task, along with the Average precision. Similarly to the training file, it can be ran directly:

```
python inference.py
```
The default parameters are the following:
- iou_thr (minimum IoU in order to be considered a true positive. The IoU is computed between the predicted and ground truth bounding boxes) --> 0.5
- nms_thr (Non Maximum Suppression threshold) --> 0.1
- model path (path to the trained model) --> "output/temp/epoch_0.pth"
- save_metrics (boolean to indicate if the user desires to store the metrics in a separate file) --> False
- dataset path (path to your data folder) --> "data_fse/"
- split (data split on which the inference will be performed) --> test
- task (inference task: detection or diameter estimation) --> "detection"
- confs (list of minimum confidence scores to iterate over) --> [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

Example: performing inference on the validation set and iterating over the minimum confidence scores of 0.1,0.3 and 0.5:
```
python inference.py --split 'val' --confs [0.1,0.3,0.5]
```
**OUTPUT:** If save_metrics is set to True, numpy arrays containing the values of P, R and F1 for each confidence level, will be stored in the directory. 

**NOTE:** The Average Precision (AP) has to be computed with the minimum confidence score (conf = 0).


### Inference on diameter estimation

The file used to compute the diameter errors is the same one as before. To do so, the task has to be indicated:

```
python inference.py --task 'diameter' 
```
**OUTPUT:** If save_metrics is set to True, two kind of results will be stored:

1) Folder with a txt per image, where each line of the txt corresponds to one detection. Each line contains: appleId|confidece score|predicted diameter|ground truth diameter|predicted visibility of the apple|ground truth visibility of the apple|x points of the mask|y points of the mask.

2) A numpy array containing all the errors of the apples considered true positives. 

**NOTE:** In the case that the save_metrics boolean is set to true, the good practice is to just iterate over one confidence score, since it will not overwrite the txt files that are already created (so the txt results will belong to the first confidece score). 


## Citation

If you find this implementation or the analysis conducted in our report helpful, please consider citing:

    @article{Farre2022,
        Author = {{Ferrer Ferrer, Mar and Ruiz-Hidalgo, Javier and Gregorio, Eduard and Vilaplana, Veronica and Morros, Josep-Ramon and Gen{\'e}-Mola, Jordi},
        Title = {Simultaneous Fruit Detection and Size Estimation Using Multitask Deep Neural Networks},
        Journal = {Submitted},
        Year = {2022}
        doi = {https://doi.org/Submitted}
    }

