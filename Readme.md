**Using tensorflow version 2.4.1**

**`Provide the dataset in the below form`**

```
sample_keras_dataset/
├── train_data/
│   ├── class1/
│   │   ├── image1.png
│   │   └── image2.png
│   └── class2/
│        ├── image1.png
│        └── image2.png
└── validation_data/
    ├── class1/
    │   ├── image1.png
    │   └── image2.png
    └── class2/
        ├── image1.png
        └── image2.png
```


`Supported architectures : `

```
VGG16
VGG19

ResNet50
ResNet101
ResNet152
ResNet101V2
ResNet152V2
ResNet50V2

NASNetLarge
NASNetMobile

MobileNet
MobileNetV2
MobileNetV3
MobileNetV3Large
MobileNetV3Small

InceptionV3

DenseNet121
DenseNet169
DenseNet201

EfficientNetB0
EfficientNetB1
EfficientNetB2
EfficientNetB3
EfficientNetB4
EfficientNetB5
EfficientNetB6
EfficientNetB7

EfficientNetV2B0
EfficientNetV2B1
EfficientNetV2B2
EfficientNetV2B3
EfficientNetV2L
EfficientNetV2M
EfficientNetV2S
```


**`HyperParameters`**

```
train_dataset_dir 
validation_dataset_dir
train_batch_size - h
class_mode
interpolation
resize_height
resize_width
include_top
final_dense
weights
feature_extractor
pretrained_model_path
distributed_strategy
pretrained_base
modelcheckpoint_callback_path
optimizer
loss
epochs
final_epoch_model_save_path
```

**`Usage`**
```
python train.py --train_dataset_dir TRAIN_DATASET_DIR --validation_dataset_dir VALIDATION_DATASET_DIR
                --train_batch_size TRAIN_BATCH_SIZE --epochs EPOCHS --validation_batch_size VALIDATION_BATCH_SIZE
                --class_mode CLASS_MODE --interpolation INTERPOLATION --input_size INPUT_SIZE --include_top INCLUDE_TOP
                --final_dense FINAL_DENSE --weights WEIGHTS --feature_extractor FEATURE_EXTRACTOR
                --pretrained_model_path PRETRAINED_MODEL_PATH --distributed_strategy DISTRIBUTED_STRATEGY
                --pretrained_base PRETRAINED_BASE --modelcheckpoint_callback_path MODELCHECKPOINT_CALLBACK_PATH
                --optimizer OPTIMIZER --loss LOSS --final_epoch_model_save_path N
```