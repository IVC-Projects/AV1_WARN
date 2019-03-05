import os, glob, re,  sys, argparse, time, random
from random import shuffle
import tensorflow as tf
import numpy as np
from PIL import Image
from WDSR8 import model
from UTILS import *

tf.logging.set_verbosity(tf.logging.WARN)

EXP_DATA = 'WDSR8_qp43_B_set2346'                  #naming model
LOW_DATA_PATH = r"D:\TRAIN_SET\AV1_NEW\qp43_B"      #The path where data is stored
HIGH_DATA_PATH = r"D:\TRAIN_SET\AV1_NEW\B_label"    #The path where label is stored
LOG_PATH = "./logs/%s/"%(EXP_DATA)            
CKPT_PATH = "./checkpoints/%s/"%(EXP_DATA)   #Store the trained models
SAMPLE_PATH = "./samples/%s/"%(EXP_DATA)      #Store result pic
PATCH_SIZE = (35, 35)        #The size of the input image in the convolutional neural network
BATCH_SIZE = 100             #The number of patches extracted from a picture added to the train set
IMAGE_BATCH = 4              #The size of mini-batch
BASE_LR = 1e-4               #Base learning rate
LR_DECAY_RATE = 0.5
LR_DECAY_STEP = 150
MAX_EPOCH = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path
#model_path = r"D:\pycharm\project\checkpoints\WDSR8_qp43_B_set2346\WDSR8_qp43_B_set2346_515.ckpt"

if __name__ == '__main__':

    train_list = get_train_list(load_file_list(LOW_DATA_PATH), load_file_list(HIGH_DATA_PATH))
    shuffle(train_list)

    with tf.name_scope('input_scope'):
        train_input = tf.placeholder('float32', shape=(BATCH_SIZE * IMAGE_BATCH, PATCH_SIZE[0], PATCH_SIZE[1], 1))
        train_gt = tf.placeholder('float32', shape=(BATCH_SIZE * IMAGE_BATCH, PATCH_SIZE[0], PATCH_SIZE[1], 1))

    shared_model = tf.make_template('shared_model', model)
    train_output, weights   = shared_model(train_input)
    #train_output = shared_model(train_input)
    train_output = tf.clip_by_value(train_output, 0., 1.)

    with tf.name_scope('loss_scope'):
        #loss = tf.reduce_mean(tf.square(tf.subtract(train_output, train_gt)))
        loss = tf.reduce_sum(tf.square(tf.subtract(train_output, train_gt)))

        # L2 norm
        for w in weights:
            loss += tf.nn.l2_loss(w) * 1e-4

        avg_loss = tf.placeholder('float32')
        tf.summary.scalar("avg_loss", avg_loss)

    global_step = tf.Variable(0, trainable=False)
    #learning_rate = tf.train.exponential_decay(BASE_LR, global_step, LR_DECAY_STEP*len(train_list), LR_DECAY_RATE, staircase=True)
    learning_rate = tf.train.exponential_decay(BASE_LR, global_step, LR_DECAY_STEP * len(train_list) // IMAGE_BATCH, LR_DECAY_RATE,
                                               staircase=True)
    tf.summary.scalar("learning rate", learning_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    opt = optimizer.minimize(loss, global_step=global_step)

    #gradient clip
    '''grads, vars = zip(*optimizer.compute_gradients(loss))
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=2.0)
    opt = optimizer.apply_gradients(zip(grads, vars), global_step=global_step)'''

    saver = tf.train.Saver(weights, max_to_keep=0)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
        if not os.path.exists(os.path.dirname(CKPT_PATH)):
            os.makedirs(os.path.dirname(CKPT_PATH))
        if not os.path.exists(SAMPLE_PATH):
            os.makedirs(SAMPLE_PATH)

        merged = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

        sess.run(tf.global_variables_initializer())

        if model_path:
            print("restore model...")
            saver.restore(sess, model_path)
            print("Done")

        #for epoch in range(516, MAX_EPOCH):
        for epoch in range(MAX_EPOCH):
            total_g_loss, n_iter = 0, 0
            idxOfImgs = np.random.permutation(len(train_list))
            epoch_time = time.time()

            for idx in range(len(idxOfImgs)// IMAGE_BATCH):
                input_data, gt_data, cbcr_data = prepare_nn_data(train_list, idx)
                feed_dict = {train_input: input_data, train_gt: gt_data}
                _, l, output, g_step= sess.run([opt, loss, train_output, global_step], feed_dict=feed_dict)
                total_g_loss += l
                n_iter += 1

                #print(output[0])
                #file_writer.add_summary(summary, g_step)
                del input_data, gt_data, cbcr_data, output
            lr, summary = sess.run([learning_rate, merged], {avg_loss:total_g_loss/n_iter})
            file_writer.add_summary(summary, epoch)
            tf.logging.warning("Epoch: [%4d/%4d]  time: %4.4f\tloss: %.8f\tlr: %.8f"%(epoch, MAX_EPOCH, time.time()-epoch_time, total_g_loss/n_iter, lr))
            #print("Epoch: [%4d/%4d]  time: %4.4f\tloss: %.8f\tlr: %.8f"%(epoch, MAX_EPOCH, time.time()-epoch_time, total_g_loss/n_iter, lr))
            '''test_out = sess.run(test_output, feed_dict={test_input: test_input_data})
            test_out = denormalize(test_out)
            save_images(test_out, test_cbcr_data, [8, 8], os.path.join(SAMPLE_PATH,"epoch%s.png"%epoch))'''

            saver.save(sess, os.path.join(CKPT_PATH, "%s_%03d.ckpt"%(EXP_DATA, epoch)))
