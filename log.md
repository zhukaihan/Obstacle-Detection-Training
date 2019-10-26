<link rel="stylesheet" href="https://github.com/sindresorhus/github-markdown-css/blob/gh-pages/github-markdown.css">

# Make Good Choices: 
We want to detect obstacles, that's why we called it obstacle detector. 

## Candidate Framework: 
 * Tensorflow
   * Pros: 
     * It is what I am familiar with, and is one of the most powerful and most used. 
     * Tensorflow Lite is available on iOS with decent optimization. 
     * Large Model Zoo. 
     * Well supported, or best supported. 
   * Cons: 
     * Kind of hard to use the API as it is not what I used to. 
 * CoreML or TuriCreate
   * Pros: 
     * It is supported by Apple, and is best supported on iOS. 
     * Super easy to use. 
   * Cons: 
     * Limited model choices. It has only YOLO v2, but v3 is better. 
     * Only available on macOS or Linux, but I uses Windows. 
 * Caffe: 
   * No no no. 

Tensorflow wins because of its large community and is available on Windows. It is a big deal because one teammmate has his RTX 2080Ti on a Windows system. 

## Candidate Model: 
 * Mask R-CNN
   * Pros: 
     * It is a scematic segmentation model as it produces masks. 
     * It is a two-step model, which is more accurate than one-shot ones. 
   * Cons: 
     * Damn slow. 1000+ ms for inference on Titan X. Imagine it is runned on iPhone. 
     * Use masks, which is a pain in the ass to create training data. 
 * YOLO (You Only Look Once)
   * Pros: 
     * You only look once!!!
     * Fast inference with decent accuracy. 
   * Cons: 
     * One-shot object detection network, not the best accuracy. 
 * SSD (Single Shot Detector)
   * Pros: 
     * One-shot object detection network, fast inference with decent accuracy. 
     * Pre-trained model already existed with lots of backbone to choose from in Tensorflow Model Zoo. 
   * Cons: 
     * Not the best accuracy. 
     * Bad accuracy to detect small objects. 

SSD has more tensorflow implementation than YOLO and pretrained weights as well. However, I will pick YOLO v3 over SSD. 

We used SSD because it is supported by Tensorflow Object Detection API and its fastness. It is a big deal because the API really shortens the development time. We used an SSD with MobileNet V2 as backbone (from the Model Zoo). 




# Therefore, the trial-and-error process is following: 
1. (DONE) I will work with SSD on Tensorflow on Windows first and see if it works. 
   1. It works pretty well... Not the best result, but still manages. 
2. (PASS) Then, I will work with modifying model's first layer to consume disparity map as the fourth channel with models in 1. 
3. (CONSIDERABLE) Then, I will work with image segmentation with Mask-RCNN in Tensorflow Object Detection API with pretrained weights to classify each pixel as road or non-road. Idea from https://stackoverflow.com/questions/6007822/finding-path-obstacles-in-a-2d-image
4. (PASS) Then, I will work with modifying model's first layer to consume disparity map as the fourth channel with models in 3. 
5. (PASS) Then, I will work on back-up plan. 




# Links: 
Tensorflow Object Detection API: https://github.com/tensorflow/models/tree/master/research/object_detection

Model Zoo: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

Data labeling software: dataturks.com

Step by Step Tensorflow Object Detection API Tutorial (step by step my ass): https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e

Image Segmentation Idea Source: https://stackoverflow.com/questions/6007822/finding-path-obstacles-in-a-2d-image




# TODOs: 

## TODOS: 
- [x] Mark some image on Dataturks and download them and convert to Pascal VOC. (Upload image, mark image, download img use script and convert to Pascal VOC use script. Should have existing scirpts)
  - Script: https://dataturks.com/help/ibbx_dataturks_to_pascal_voc_format.php

- [x] Install Tensorflow Object Detection API and convert Pascal VOC to TFRecord. (complicate installation and dataset prep and training and testing and exporting) 
  - Script: https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py
  - Script Explanation: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md
  - Tutorial: Shit tutorial. Tells you that there is script and done. Existing script works for official Pascal VOC but not our structure of folders. 

- [x] Use existing SSD Tensorflow implementation with Tensorflow Object Detection API. (Finished dataset prep, tried to train a model) 
  - Training pipeline: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md

- [x] Use Tensorflow Lite and the trained SSD model on iPhone. 

- [ ] Build a UI. 

- [ ] Train another model at 500 images. 

- [ ] Train another model at 750 images. 

- [ ] Train another model at 1000 images. 

## Postponed: 
- [ ] As an overachiever, use sematic segmentation, with Mask-RCNN or DeepLab, with Tensorflow Object Detection API, with pretrained weights, with mask data. 

## Postponed Indefinitely: 
- [ ] Add 4th layer to input layer of SSD. 
- [ ] Use existing YOLO Tensorflow implementation to try out YOLO, both in training and testing and on iOS. https://github.com/hizhangp/yolo_tensorflow https://github.com/mystic123/tensorflow-yolo-v3 (complicated dataset, complicated training and testing and exporting)
- [ ] Use Turicreate, create dataset for Turicreate and train a model. (figure out dataset and training and exporting is simple)
  - Buy a new Mac first. 



---
<div style="position: relative; left: 0; width: 100%; margin: 0; padding-bottom: 50%; padding-top: 50%;">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(0deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(10deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(20deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(30deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(40deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(50deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(60deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(70deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(80deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(90deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(100deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(110deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(120deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(130deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(140deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(150deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(160deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(170deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(180deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(190deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(200deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(210deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(220deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(230deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(240deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(250deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(260deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(270deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(280deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(290deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(300deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(310deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(320deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(330deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(340deg);">
<hr style="position: absolute; left: 0; width: 100%; margin: 0; transform: rotate(350deg);">
</div>

<div style="font-size:50px;">Guides Coming Up: </div>
---



# Install Tensorflow Object Detection API: 

The procedure is simple. Follow the guide. 

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

However, since I am installing on Windows, a few tweeks shall take place. 

 1. Dependencies: 
    * Anaconda should have all the dependencies installed. I only need to install pillow (through conda as well). 
    * Although Cython is installed, Visual C++ Build Tools from Microsoft is still needed. 
    * When installing Tensorflow through conda, conda will have its own CUDA and cuDnn. Its wierd but make sure you have installed CUDA and cuDnn through NVIDIA's instructions, and have versions up to date. 
 2. COCO API installation
    * It seems to be Linux only. But, all it does is a make command, which we can look into its Makefile to see that it only does a python setup. 
    * Make sure Visual C++ Build Tools is installed. Not the distribution, but the build tools. It can be installed as standalone, as well as through Visual Studio. 
    * setup.py needs to be modified as it contains Linux gcc flags. See here: https://github.com/cocodataset/cocoapi/issues/51#issuecomment-379872704. 
 3. Protobuf Compilation
    * Just use a prebuilt binary for Windows. 
 4. Add Libraries to PYTHONPATH
    * Open up environment variable settings on Windows, add PYTHONPATH...




# Download Data and Prepare Data: 


## In General: 

### Story: 
What I did was to edit the scripts from dataturks to pascal voc and from pascal voc to tensorflow record. 

I thought about combining both scripts into one. However, the script tensorflow provides uses tensorflow as intermediate between console and actual script. Therefore, using two scripts does not seems to be too much of a big deal. 

### Scripts: 
 * mkdir_and_random_train_val.py: Randomly chooses train and val data. 
 * dataturks_to_pascal.py: Converts dataturks format into Pascal VOC format. It downloads images as well. Modified from original script from https://dataturks.com/help/ibbx_dataturks_to_pascal_voc_format.php. 
 * pascal_to_tf.py: Converts Pascal VOC format to TFRecord. Modified from https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py. 

### Data Preparation General Pipeline: 
 1. Upload data onto dataturks.com and label each image. 
 2. Download the json file that has the url of each uploaded image and its annotation. 
 3. Manually write label_map.pbtxt. It matches different labels to a number. This is simple. 
 4. Use mkdir_and_random_train_val.py, to read in the json file for a list of all images and randomly choose train and validation data. The result will be stored in train.txt and val.txt, which will be needed for converting Pascal VOC to TFRecord. The train.txt and val.txt has one image name per line, with " 1" following the name. I have not yet discovered what that 1 or -1 is for in PASCAL VOC format, but this number is not used in pascal_to_tf.py. This script will also create folders for storing images and annotations. 
 5. Use dataturks_to_pascal.py to download the data. The script is written for Linux, thus, it parses paths with "/". However, I am using Windows. Therefore, there are a few places that needs to change "/" to "\". Otherwise, the image will download in a wierd file structure. 
 6. Use pascal_to_tf.py to convert Pascal VOC format to TFRecord. What this script is different from original one is that there are fields that dataturks does not support. These fields are removed from conversion. The script is also altered to fit our file structure. 
 7. Done. You should have train.record and val.record. 

The json file downloaded from dataturks will be called "Obstacle Detection Dataset.json"


## Step 1: Create Label Map: 

```
item {
  id: 1
  name: 'Obstacle'
}
item {
  id: 2
  name: 'Pothole'
}
......etc......
```

There is no id 0. id of 0 is a placeholder. Therefore, id starts from 1. 


## Step 2: Make Data Folder Structure for Conversions and Randomly Select Train and Val Data: 

```bash
python mkdir_and_random_train_val.py
```

Directories are defined in mkdir_and_random_train_val.py file. 

It will create a directory: 

```
+formatted data
  +annotated
  +imgs
```

It will also create two files: train.txt and val.txt. 
 * train.txt: Image names for training set. 
 * val.txt: Image names for validation set. 


## Step 3: Convert From Dataturks to Pascal VOC (Also Downloads Images): 

```bash
python dataturks_to_pascal.py "Obstacle Detection Dataset.json" "formatted data\imgs" "formatted data\annotated"
```

Modified from https://dataturks.com/help/ibbx_dataturks_to_pascal_voc_format.php. 

Beaware of difference between Linux and Windows path representation. Linux uses '/' while Windows uses '\'. 


## Step 4: Convert From Pascal VOC to TFRecord: 

```bash
python pascal_to_tf.py --data_dir="formatted data" --annotations_dir="formatted data\annotated" --output_path=pascal.record --label_map_path=label_map.pbtxt --set=train
```

set parameter can be "train" or "val", which it will read train.txt or val.txt to create training set or val set. 

Modified from models-master\research\object_detection\dataset_tools\create_pascal_tf_record.py

Beaware of difference between Linux and Windows path representation. Linux uses '/' while Windows uses '\'. 




# Fine-tune SSD: 

We have data processed. We need to train SSD to fit out data. 

Guide is here: 
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md. 


## Step 1: Create Training Folder Structure: 

The folder structure is like this: 

```
+data
  -label_map file
  -train TFRecord file (train.record)
  -eval TFRecord file (val.record)
+models
  + model
    -pipeline config file (pipeline.config from pretrained model)
    -checkpoint files (model.ckpt.data-00000-of-00001 model.ckpt.index model.ckpt.meta)
    +train
    +eval
```


## Step 2: Modify pipeline.config: 

There is one line in SSD's pipeline.config (a line about batch_norm) that will not pass. Simply delete that line. 

Modify the paths to something like this: 

```
model: 
  fine_tune_checkpoint: "C:/Users/zhuka/iCloudDrive/Desktop/Obstacle Detection/train/models/model/
train: 
    label_map_path: "C:/Users/zhuka/iCloudDrive/Desktop/Obstacle Detection/train/data/label_map.pbtxt"
    input_path: "C:/Users/zhuka/iCloudDrive/Desktop/Obstacle Detection/train/data/train.record"
val: 
    label_map_path: "C:/Users/zhuka/iCloudDrive/Desktop/Obstacle Detection/train/data/label_map.pbtxt"
    input_path: "C:/Users/zhuka/iCloudDrive/Desktop/Obstacle Detection/train/data/val.record"
```

Absolute path seems necessary. Change them accordingly. 

Beaware of difference between Linux and Windows path representation. Linux uses '/' while Windows uses '\'. 


## Step 3: Training: 

```bash
PIPELINE_CONFIG_PATH={path to pipeline config file}
MODEL_DIR={path to model directory}
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python model_main.py --pipeline_config_path=${PIPELINE_CONFIG_PATH} --model_dir=${MODEL_DIR} --num_train_steps=${NUM_TRAIN_STEPS} --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES --alsologtostderr
```

model_main.py located in tensorflow/models/research/object_detection directory

Change these variables to suites your needs. 


## Step 3.5: Saving: 

By default, the model saves at most 5 checkpoints. To find the optimal checkpoint, we need to save the new checkpoint somewhere else. This process is not user-friendly since it is manual. 

Therefore, I changed saving checkpoints to 100 by modifying following code: 

In object detection/model_lib.py, we modify line ~490, which creates a saver. 
```python
saver = tf.train.Saver(
    sharded=True,
    keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
    save_relative_paths=True,
    max_to_keep=10000)
```

This change suffice our needs, but to keep everything consistent, we shall change line ~470 in the same file. 

We hardcode this value because it is simple to do. If we want to not to hardcode it in our code, we need to put it in pipeline.config, edit pipeline.proto, recompile proto, and then it may work. 

I have also modified line 62 in model_main.py to: 
```python
config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir, keep_checkpoint_max=10000)
```
Setting this only would not work. I am not sure if not setting it and only sets the saver would work or not. 


## Step 4: Open TensorBoard: 

```bash
tensorboard --logdir=${MODEL_DIR}
```




# Export Model to TFLite: 


## Step 1: Convert from checkpoints to tflite.pb: 

```bash
python export_tflite_ssd_graph.py \
--pipeline_config_path=pipeline.config \
--trained_checkpoint_prefix=model.ckpt-16812\
--output_directory=output_tflite \
--add_postprocessing_op=true
```

trained_checkpoint_prefix should matches three files: .data-00000-of-00001 .index .meta. 

Using this script is to make sure that all operations from tflite.pb will be supported in Tensorflow Lite. 


## Step 2a: Convert from tflite.pb to .tflite files (Non Quantized Model): 

```bash
tflite_convert --output_file=ssd.tflite --graph_def_file=output_tflite\tflite_graph.pb --input_arrays=normalized_input_image_tensor --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 --input_shapes=1,300,300,3 --allow_custom_op
```

input_shape, input_arrays, and output_arrays are described in export_tflite_ssd_graph.py file header. 

output_arrays is TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 because TFLite_Detection_PostProcess tensor has 4 outputs. TFLite_Detection_PostProcess alone will only includes the first output. It is similar to pointer array (CSE 30~~~, thanks, Rick). 

TFLite_Detection_PostProcess is a custom operation, thus adding allow_custom_op is necessary because tflite_convert thought native TFLite will not support it. However, it is actually implemented. Therefore, we need to bypass this check. There is no other custom operation in the network. 

input_shape's width and height is defined in pipeline.config: 

```
fixed_shape_resizer {
    height: 300
    width: 300
}
```

Original Guide is here: https://www.tensorflow.org/lite/convert/cmdline_examples#convert_a_tensorflow_graphdef_. 


## Step 2b: Convert from tflite.pb to .tflite files (Quantized Model): 

```bash
tflite_convert --output_file=model_quantized.tflite --graph_def_file=output_tflite\tflite_graph.pb --input_arrays=normalized_input_image_tensor --output_arrays=raw_outputs/box_encodings,raw_outputs/class_predictions --input_shapes=1,300,300,3 --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_dev_values=127 --default_ranges_min=0 --default_ranges_max=6
```

We use dummy quantization because the trained model may have layers that does not have max and min values for quantization. For example, Relu6 does not have such information. Therefore, we are specifying plausible min and max. 

Original Guide is here: https://www.tensorflow.org/lite/convert/cmdline_examples#use_dummy-quantization_to_try_out_quantized_inference_on_a_float_graph_. 




# Deploy: 


Guide is here: https://www.tensorflow.org/lite/guide/ios. 

This guide lets talks about how to run an example project and how to understand the code. 


## Step 1: Install iOS Tensorflow Lite in Xcode Project: 

The Tensorflow Lite is installed through CocoaPods. 

Install cocoapods. 

In Podfile under the project folder: 

```
target 'Obstacle-Detection' do
    pod 'Zip', '~> 1.1'
    #pod 'TensorFlow-experimental', '~> 1.1'
    #pod 'TensorFlowLiteGpuExperimental', '0.0.1'
end
```

Zip is another framework Obstacle-Detection used. 

Choose either TensorFlow-experimental or TensorFlowLiteGpuExperimental. 
If you want to use GPU to run the network, install TensorFlowLiteGpuExperimental. 
To be mindful, there are extra steps to configure the project if you want to use GPU. Read:
https://www.tensorflow.org/lite/performance/gpu. In this tutorial, TFLITE_USE_CONTRIB_LITE does not exist in sample code, simply ignore. 

Then, in bash under that project folder: 

```bash
pod install
```

Or have installed before, use: 

```bash
pod repo update
```




## Step 2: Develop iOS App: 

Just use the camera example from tensorflow lite, and modify elemtents. 

Click into TFLite methods to see its description. 

1. Copy the converted model over. 
   1. Copy the model to the project. 
   2. Add the model to Bundle Resources. 
2. List the labels used. 
3. Change parameters of model and its input. 
4. Change some code in runModelOnFrame() to handle the output of the model. 

Bemindful that there is an input process error in the demo code, the x and y values are flipped. Code has to be modified for both quantized and non-quantized functions. Details are here: 
https://github.com/tensorflow/tensorflow/issues/25784

Bemindful that the demo code is missing an input check. In ODModelEvaluator.mm, function evaluateOnBuffer, it calls CFRetain on pixelBuffer directly without checking whether pixelBuffer is NULL or not. The pixelBuffer returned by CMSampleBufferGetImageBuffer can be NULL. CFRetain(NULL) will crash the app. Thus, add a condition to check whether pixelBuffer is NULL or not is necessary before calling CFRetain(pixelBuffer). 





---
<div style="font-size:50px;">Results and a Conclusion Coming Up: </div>
---

The initial model that was trained on 45 images was having decent recognition on big boxes and poles. Even a slim coke bottle is recognized as an obstacle. Therefore, we can conclude that our architecture can converge. 

The complete dataset was then used. After early stopping technique was used, the model does not recognize well, worse than the initial model. With an overfitted model, the model does not recognize at all. 

It turns out, the data was poorly labeled such that some boxes are not tight, some obstacles are not drawn, some edges are labeled with multiple boxes, some boxes are not bounding the object, etc. 

After all of these problems were solved, We are able to achieve a decent detection. Our peak mAP is 0.15, peak mAP@.50IOU is 0.29, and mAP large is 0.18. Recall that we only have 315 images. 

There is an interesting observation: Fire hydrant is particularily accurate as a fire hydrant in pretrained model, and is also super accurate as an obstacle. However, when I remove the blue channel, the pretrained model categorized it as a bottle... 

From such observation, I believe we can reach this result with such limited images can be due to the fact that obstacles are a superset of lots of objects. The pretrained model has been trained on lots and lots of objects. Therefore, during the fine-tuning process, our model does not need to learn new features or new objects, but only need to categorize the ones that exists. This could be the reason why we are able to obtain such accuracy with a limited dataset. 

According to Apple, some model can be fine-tuned with as little as 60 images... But, as overachievers, we are looking to expand our dataset to at least 1000 images. 

