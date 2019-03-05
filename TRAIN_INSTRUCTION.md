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
* training/ResNet8.py : 
* training/evaluate.py : test the generalization power of the saved checkpoints.

## Environment setup

The following packages are required:
* TensorFlow
* Python
* pillow

## Dataset
To start training model, first download the dataset and extract it into the ./data directory.  在线学习，mini-batch权衡trade off效率和显存,
The images in the dataset are all in the YUV format.

## Train method
To train the model, the initial learning rate is set to 0.0001. The learning rate is adjusted with the step strategy using gamma=0.5. In our implementation, the learning rate is multiplied by 0.5 every 180 epochs in QP=52 for the intra coding. In terms of the inter coding, the learning rate is halved per 80 epochs in QP=52. And, Small QP may converge faster.


* WDSR models can be trained with a pixel-wise loss function with NN_RUN.py.Default for WDSR is mean squared error.<br>
For example,

        python NN_RUN.py

    VDSRx15_SE_qp37.log is the file you want to redirect the output to, which you can change at will.

* if start with a checkpoint

        python VDSRTEST.py --model_path=./checkpoints/CHECKPOINT_NAME.ckpt

* The commands for launching tensorboard:

        tensorboard --logdir=logs
        
    The `--logdir` option sets the location point to the log directory of the job.If you still have difficulty in launching tensorboard,you can refer to the https://www.jianshu.com/p/d8f9b0dfacdb.

## Evaluation

to evaluate saved models with evaluate.py and then select the model with the highest PSNR. For example,

        python VDSRTEST.py >>./train_log/VDSRx15_SE_qp37_Test.log 2>&1

## FAQ


