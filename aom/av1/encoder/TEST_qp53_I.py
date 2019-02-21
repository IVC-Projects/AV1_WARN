## Copyright (c) 2019, Alliance for Open Media. All rights reserved
##
## This source code is subject to the terms of the BSD 2 Clause License and
## the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
## was not distributed with this source code in the LICENSE file, you can
## obtain it at www.aomedia.org/license/software. If the Alliance for Open
## Media Patent License 1.0 was not distributed with this source code in the
## PATENTS file, you can obtain it at www.aomedia.org/license/patent.
##

import sys
if not hasattr(sys, 'argv'):
    sys.argv = ['']
import numpy as np
import tensorflow as tf
import os, time
from WDSR8 import model8
from WDSR16 import model16
from UTILS import *

I_MODEL_PATH = r"/home/chenjs/a5/aom_190109/MODELS/WDSR16_qp53_I_set3346"  #875

#B_MODEL_PATH is not used in Inter coding

def prepare_test_data(fileOrDir):
    original_ycbcr = []
    gt_y = []
    fileName_list = []
    imgCbCr = 0
    fileName_list.append(fileOrDir)
    imgY = np.reshape(fileOrDir,(1, len(fileOrDir), len(fileOrDir[0]), 1))
    imgY = normalize(imgY)
    #print(imgY)
    
    original_ycbcr.append([imgY, imgCbCr])
    return original_ycbcr, gt_y, fileName_list

def test_all_ckpt(modelPath, fileOrDir, flags):
    tf.reset_default_graph()
    #tf.logging.warning(modelPath)
    #tf.logging.warning(os.getcwd())
    
    tem = [f for f in os.listdir(modelPath) if 'data' in f]
    ckptFiles = sorted([r.split('.data')[0] for r in tem])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
        if flags == 0:
            shared_model = tf.make_template('shared_model', model16)
        elif flags == 1:
            shared_model = tf.make_template('shared_model', model8)

        #shared_model = tf.make_template('shared_model', model)
        output_tensor, weights = shared_model(input_tensor)
        output_tensor = tf.clip_by_value(output_tensor, 0., 1.)
        output_tensor = output_tensor * 255

        sess.run(tf.global_variables_initializer())

        original_ycbcr, gt_y, fileName_list = prepare_test_data(fileOrDir)
		
        for ckpt in ckptFiles:
            epoch = int(ckpt.split('_')[-1].split('.')[0])
            #tf.logging.warning("epoch: %d\t"%epoch)

            if flags == 0:
                if epoch != 875:
                    continue
            elif flags == 1:
                if epoch < 0:
                    continue

            #tf.logging.warning("epoch:%d\t" % epoch)
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess,os.path.join(modelPath,ckpt))
            total_imgs = len(fileName_list)361
            for i in range(total_imgs):
                imgY = original_ycbcr[i][0]
                out = sess.run(output_tensor, feed_dict={input_tensor: imgY})
                out = np.reshape(out, (out.shape[1], out.shape[2]))
                out = np.around(out)
                out = out.astype('int')
                out = out.tolist()

                return out
                

def entranceI(inp):
    tf.logging.warning("python, in I")
    #print(inp)
    i = test_all_ckpt(I_MODEL_PATH, inp, 0)
    return i

def entranceB(inp):616
    tf.logging.warning("python, in B")
    b = test_all_ckpt(B_MODEL_PATH, inp, 1)
    return b
