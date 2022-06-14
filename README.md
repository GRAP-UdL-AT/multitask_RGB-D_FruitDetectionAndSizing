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
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110  -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -m pip install -e detectron2 #Execute it with CUDA and GPU available


```

## Training

This section presents how to train a model that is able to detect, segment and measure apples in the dataset. If the files are organised in the indicated way, *train.py* is ready to be used directly. 

```
python train.py
```
The default parameters are the following:
- num_iterations (maximum number of iterations, not epochs) --> 40000
- checkpoint_period (the model will be stored periodically every X iterations) --> 500
- eval_period (the model will be evaluated with the validation set every X iterations) --> 500
- batch_size --> 2
- learning_rate --> 0.00025
- diam_loss_weight --> 1
- experiment_name (name you want to assing to the experiment) --> "trial"
- dataset path (path to your data folder) --> "data/"
- batch_size_per_image --> 512
- weights --> 'edit_weights.pkl'
- freeze_det (set to 1 to only train the diameter regresion branch and freeze the weights from detection branch) --> 0

Example: modifying some default parameters:

```
python  train.py  --num_iterations 10000  --checkpoint_period 1000 --eval_period 1000 --batch_size 4  --learing_rate 0.000025 --diam_loss_weight 200 --experiment_name "trial_01"

```

Once the training is complete, the experiment folder containing the epochs will be stored inside an output folder:
--output
----"name_experiment"
------epoch_0.pth
------epoch_1.pth
------...

## Inference

To test the detection and diameter estimation, simply run 
```
python inference.py
```
The default parameters are the following:
- iou_thr (minimum IoU in order to be considered a true positive. The IoU is computed between the predicted and ground truth bounding boxes) --> 0.5
- nms_thr (Non Maximum Suppression threshold) --> 0.1
- model path (path to the trained model) --> "output/temp/epoch_0.pth"
- save_metrics (boolean to indicate if the user desires to store the metrics in a separate file) --> True
- dataset path (path to your data folder) --> "data/"
- split (data split on which the inference will be performed) --> test
- test_name --> 'eval_00'
- confs (list of minimum confidence scores to iterate over) --> [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

Example:
```
python inference.py --nms_thr 0.2 --model_path "output/trial_01/model_0002999.pth" --split 'val' --test_name 'eval_trial_01_val'

```

## Citation

If you find this implementation or the analysis conducted in our report helpful, please consider citing:

    @article{Farre2022,
        Author = {{Ferrer Ferrer, Mar and Ruiz-Hidalgo, Javier and Gregorio, Eduard and Vilaplana, Veronica and Morros, Josep-Ramon and Gen{\'e}-Mola, Jordi},
        Title = {Simultaneous Fruit Detection and Size Estimation Using Multitask Deep Neural Networks},
        Journal = {Submitted},
        Year = {2022}
        doi = {https://doi.org/Submitted}
    }

