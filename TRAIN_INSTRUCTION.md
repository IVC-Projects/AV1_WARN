# The training code, method and instructions

## Table of contents

- [Files](#Files)
- [Environment setup](#Environment-setup)
- [Dataset](#Dataset)
- [Training method](#Training-method)
- [Evaluation](#Evaluation)
- [FAQ](#FAQ)

## Files

* training/train.py : main training file.
* training/UTILS.py : defines some functions that need to be used in the training process, such as how to load the data set and how to calculate PSNR.
* training/WDSR8.py : A TensorFlow-based implementation of [Wide Activation for Efficient and Accurate Image Super-Resolution](https://arxiv.org/abs/1808.08718) (WDSR), winner 
  of the [NTIRE 2018](http://www.vision.ee.ethz.ch/ntire18/) super-resolution challenge.<br>
 We adopted Xavier Initialization as the method for weight initialization.
* training/ResNet8.py : plain residual network with 8 residual blocks(ResNet).
* training/evaluate.py : test the generalization power of the saved checkpoints.

## Environment setup

The following packages are required:
* TensorFlow 1.x
* Python 3.x
* pillow

## Dataset

To start training model, first download the dataset and extract training data and label into two directories separately. Also, you need to modify the `LOW_DATA_PATH` and `HIGH_DATA_PATH` values in train.py to point to them separately.<br>
Note: The images in the dataset are all in the YUV format. In addition, the names of the images contain information about heights and widths of the images in a uniform format.For example, the  size of the image is recorded at index 3 position if `out_0000_BasketballDrive_1920x1080_50` file is split by `'_'`.

## Training method

To train the model, the initial learning rate is set to 0.0001 and parameters β1 = 0.9, β2 = 0.999. The learning rate is adjusted with the step strategy using gamma=0.5. In our implementation, the learning rate is halved every 150 epochs in QP=43 for the intra coding. 
All networks were trained using the Adam optimizer,and 400 examples in each iteration.<br>

* Firstly, you need to modify the extraction rules in the `getWH(yuvfileName)` function in training/UTILS.py, which extracts the height and width of images in yuv format from their names.

* Make sure the path of training set is correct. Then, WDSR models can be trained with a pixel-wise loss function with train.py. Default for WDSR is mean squared error. For example,

        python train.py >>FILE_NAME.log 2>&1

    FILE_NAME.log is the file you want to redirect the output to.

* if start with a checkpoint,

        python train.py --model_path=./checkpoints/CHECKPOINT_NAME.ckpt

* In order to view stats during training (image previews, scalar for loss), simply run

        tensorboard --logdir=logs
        
    The `--logdir` option sets the location point to the log directory of the job.

## Evaluation

To evaluate saved models with evaluate.py and then select the epoch with the highest PSNR. For example,

        python evaluate.py >>./train_log/FILE_NAME.log 2>&1

The `EXP_DATA` in evaluate.py parameter indicates the model you want to evaluate. Similarly, you need to update the `ORIGINAL_PATH` and `GT_PATH` values in evaluate.py to point to your validation set. Pay attention to the name format of the validation set images.

## FAQ
1.You might get a slight error running evaluation.py to evaluate the generalization abilities of the models. The main reason is probably something wrong with the directory for the low-pixel images in evaluation.py. More precisely, the current directory in which the low-pixel images are located is not named like 'QP63' or 'QP53' or anything like that. Because it is required to subtract the PSNR in AV1 anchor from the PSNR of the images through CNN to calculate the gain and find the best model at the end of the evaluation.py file.

