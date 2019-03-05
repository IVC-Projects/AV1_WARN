# The training code, method and instructions

## Table of contents

- [Files](#Files)
- [Environment setup](#Environment-setup)
- [Dataset](#Dataset)
- [Train method](#Train-method)
- [Evaluation](#Evaluation)
- [FAQ](#FAQ)

## Files

* training/train.py : main training file.
* training/UTILS.py : defines some functions that need to be used in the training process, such as how to load the data set and how to calculate PSNR.
* training/WDSR8.py : A TensorFlow-based implementation of [Wide Activation for Efficient and Accurate Image Super-Resolution](https://arxiv.org/abs/1808.08718) (WDSR), winner 
  of the [NTIRE 2018](http://www.vision.ee.ethz.ch/ntire18/) super-resolution challenge.
* training/ResNet8.py : plain residual network with 8 residual blocks(ResNet).
* training/evaluate.py : test the generalization power of the saved checkpoints.

## Environment setup

The following packages are required:
* TensorFlow
* Python
* pillow

## Dataset

To start training model, first download the dataset and extract data and training label into two directories separately. Also, you need to modify the `LOW_DATA_PATH` and `HIGH_DATA_PATH` values in train.py to point to them separately.<br>
Note:The images in the dataset are all in the YUV format.

## Train method
To train the model, the initial learning rate is set to 0.0001 and parameters β1 = 0.9, β2 = 0.999. The learning rate is adjusted with the step strategy using gamma=0.5. In our implementation, the learning rate is halved every 150 epochs in QP=43 for the intra coding. 
All networks were trained using the Adam optimizer,and 400 examples in each iteration.

* WDSR models can be trained with a pixel-wise loss function with train.py.Default for WDSR is mean squared error.For example,

        python train.py >>FILE_NAME.log 2>&1

    FILE_NAME.log is the file you want to redirect the output to.

* if start with a checkpoint

        python VDSRTEST.py --model_path=./checkpoints/CHECKPOINT_NAME.ckpt

* The commands for launching tensorboard:

        tensorboard --logdir=logs
        
    The `--logdir` option sets the location point to the log directory of the job.If you still have difficulty in launching tensorboard,you can refer to the https://www.jianshu.com/p/d8f9b0dfacdb.

## Evaluation

to evaluate saved models with evaluate.py and then select the model with the highest PSNR. For example,

        python evaluate.py >>./train_log/FILE_NAME.log 2>&1

## FAQ


