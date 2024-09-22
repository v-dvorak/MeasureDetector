# Measure Detector

This project is based on [Alexander Pacha's Measure Detector (2019)](https://github.com/OMR-Research/MeasureDetector), adapted to work in 2024 it is a part of [my OMR research](https://github.com/v-dvorak/omr-layout-analysis) (OMRLA). This fork does not necessarily fix all the issues, only the ones that are crutial to run model training.

Models trained in this project can detect: system measures, stave measures, staves, systems and grand staves. Based on the Tensorflow Object Detection API.

## Quick Start

- [Use the model (inference)](MeasureDetector/demo/)
- [Dataset overview (OMRLA)](https://github.com/v-dvorak/omr-layout-analysis?tab=readme-ov-file#dataset-overview)

## Disclamer

This project requires TensorFlow 1.13.1 that is no longer maintained, to run it you need to use Python 3.6 or 3.7 which are both heavily outdated and reached their end-of-line. If you would like to base anything on this project I would recommend to work with current version of Python and TensorFlow 2.x. I use these libraries only to replicate Pacha's work.

This code has only been tested on Linux.

## Install required libraries

- Python 3.6 or 3.7

```bash
# from MeasureDetector/
pip install -r requirements.txt
```

### Adding source to Python path

To permanently link the source-code of the project, for Python to be able to find it, you can link the two packages by running:
```bash
# From MeasureDetector/
pip install -e .
# From MeasureDetector/research/
pip install -e .
cd slim
# From inside MeasureDetector/research/slim
pip install -e .
```

### Build Protobuf 
Tensorflow Object Detection API requires to build a few python files from underlying protobuf-specifications. To do so, we run the Protobuf-Compiler:

```bash
# Install Protobuf-Compiler, e.g.,
sudo apt install protobuf-compiler
# From MeasureDetector/research/
protoc object_detection/protos/*.proto --python_out=.
```

## Datasets

Datasets used to train the model are listed [here](https://github.com/v-dvorak/omr-layout-analysis?tab=readme-ov-file#dataset-overview). They are transformed from the COCO format used in the other project to COCO format used in this one - the main difference being the format `(left, top, width, height)` vs `(left, top, right, bottom)`.

## Prepare datasets for training

These instructions are for datasets provided by the OMRLA project only.
First, we need to retrieve the annotations and transform them, the script can handle multiple datasets at once.

```bash
# From anywhere
python build_dataset.py /final/dataset/dir path/to/dataset1 path/to/dataset2
```

```bash
# From MeasureDetector/MeasureDetector
python create_joint_dataset_annotations.py --dataset_directory /final/dataset/dir --split train/test split ratio as float in [0.0, 1.0]
```

This will create four files: A json file with all annotations and three distinct files which contain random splits for training, validation and test. But for the purpose of this project we only use train and test files.

Finally you can take any of these json-files to create the TF-Record file that will be used for training:
 
```bash
# From MeasureDetector/MeasureDetector
python create_tf_record_from_joint_dataset.py \
  --dataset_directory MeasureDetectionDatasets \
  --annotation_filename training_joint_dataset.json \
  --output_path MeasureDetectionDatasets\training.record \
  --target_size=4000

python create_tf_record_from_joint_dataset.py \
  --dataset_directory MeasureDetectionDatasets \
  --annotation_filename test_joint_dataset.json \
  --output_path MeasureDetectionDatasets\test.record \
  --target_size=800
```

> For this project we do not allow `sample_reuse`. Alway pass precise `target_size` to the script, if the number is higher than the actual number of samples, the script will get stuck in an infinite loop.

## Running the training

Before starting the training, you need to change the paths in the configuration, you want to run, e.g. `configurations/faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained_reduced_classes.config`

To start the training:

```bash
# From MeasureDetector/
python research/object_detection/legacy/train.py \
  --pipeline_config_path="MeasureDetector/configurations/faster_rcnn_inception_v2_all_datasets.config" \
  --train_dir="data/faster_rcnn_inception_v2_all_datasets"
```

To start the validation: 

```bash
# From MeasureDetector/
python research/object_detection/eval.py \
  --pipeline_config_path="MeasureDetector/configurations/faster_rcnn_inception_v2_all_datasets.config" \
  --checkpoint_dir="data/faster_rcnn_inception_v2_all_datasets" \
  --eval_dir="data/faster_rcnn_inception_v2_all_datasets/eval"
```

A few remarks: The two scripts can and should be run at the same time, to get a live evaluation during the training. The training progress can be visualized by calling `tensorboard --logdir=checkpoint-directory`.

## Training with pre-trained weights

It is recommended that you use pre-trained weights for known networks to speed up training and improve overall results. To do so, head over to the [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md), download and unzip the respective trained model, e.g. `faster_rcnn_inception_resnet_v2_atrous_coco` for reproducing the best results, we obtained. The path to the unzipped files, must be specified inside of the configuration in the `train_config`-section, e.g.,

```
train-config: {
  fine_tune_checkpoint: "C:/Users/Alex/Repositories/MusicObjectDetector-TF/MusicObjectDetector/data/faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08/model.ckpt"
  from_detection_checkpoint: true
}
```

> Note that inside that folder, there is no actual file, called `model.ckpt`, but multiple files called `model.ckpt.[something]`.

## Inference

Once, you have a trained model that you are happy with, you can freeze it for deployment and usage for inference.

```bash
# From MeasureDetector/
python research/object_detection/export_inference_graph.py \
   --input_type image_tensor \
   --pipeline_config_path "MeasureDetector/data/faster_rcnn_inception_resnet_v2_atrous_muscima_pp_fine_grid/pipeline.config" \ 
   --trained_checkpoint_prefix "MeasureDetector/data/faster_rcnn_inception_resnet_v2_atrous_muscima_pp_fine_grid/model.ckpt-3814" \ 
   --output_directory "MeasureDetector/data/faster_rcnn_inception_resnet_v2_atrous_muscima_pp_fine_grid/output_inference_graph"
```

To run inference, see the [Demo](MeasureDetector/demo/) folder for more information. 

## Common issues

### Script crashes with `BadArgument` error

Check if paths passed to scripts are ok. This may be caused by `gfile.Copy` method provided by TensorFlow. To my knowledge, this method is outdated and only copies file from A to B and fails even if the paths provided are correct and exist.

Remove any usage of this method from code affected. Most of the time, this method is used to store config info within a model folder. To make sure you store config files for every training run, replace it with `shutil`'s `copy2` method.

## References

[Alexander Pacha](http://alexanderpacha.com), [TU Wien](https://www.ims.tuwien.ac.at/people/alexander-pacha) and [Simon Waloschek, Detmold University of Music](http://www.cemfi.de/people/simon-waloschek), 2019, MeasureDetector, available online at [GitHub](https://github.com/OMR-Research/MeasureDetector)
