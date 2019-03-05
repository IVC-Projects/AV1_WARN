import numpy as np
from PIL import Image
import tensorflow as tf
import os, time
from WDSR8 import model
from UTILS import *

tf.logging.set_verbosity(tf.logging.WARN)

EXP_DATA = "WDSR8_qp43_B_set2346"
TESTOUT_PATH = "./testout/%s/"%(EXP_DATA)
MODEL_PATH = "./checkpoints/%s/"%(EXP_DATA)
ORIGINAL_PATH = r"D:\pycharm\project\TEST\av1_test_after_deblock_1221\QP43"  #Low resolution images
GT_PATH = r"D:\pycharm\project\TEST\av1_test_gt"       #The original images(Ground truth)
OUT_DATA_PATH = "./outdata/%s/"%(EXP_DATA)
#The average PSNR of the validation set after intra coding of AV1
#NOFILTER = {'QP32':37.434, 'QP43':33.849, 'QP53':30.696, 'QP63':26.132}
#The average PSNR of the validation set after Inter coding of AV1
NOFILTER = {'QP32':38.304, 'QP43':35.3085, 'QP53':32.827, 'QP63':29.078}

##Ground truth images dir should be the 2nd component of 'fileOrDir' if 2 components are given.

##cb, cr components are not implemented
def prepare_test_data(fileOrDir):
    if not os.path.exists(TESTOUT_PATH):
        os.makedirs(TESTOUT_PATH)
    if not os.path.exists(OUT_DATA_PATH):
        os.makedirs(OUT_DATA_PATH)

    original_ycbcr = []
    gt_y = []
    fileName_list = []
    #The input is a single file.
    if type(fileOrDir) is str:
        fileName_list.append(fileOrDir)

        #w, h = getWH(fileOrDir)
        #imgY = getYdata(fileOrDir, [w, h])
        imgY = c_getYdata(fileOrDir)
        imgY = normalize(imgY)

        imgY = np.resize(imgY, (1, imgY.shape[0], imgY.shape[1], 1))
        original_ycbcr.append([imgY, imgCbCr])

    ##The input is one directory of test images.
    elif len(fileOrDir) == 1:
        fileName_list = load_file_list(fileOrDir)
        for path in fileName_list:
            #w, h = getWH(path)
            #imgY = getYdata(path, [w, h])
            imgY = c_getYdata(path)
            imgY = normalize(imgY)

            imgY = np.resize(imgY, (1, imgY.shape[0], imgY.shape[1], 1))
            original_ycbcr.append([imgY, imgCbCr])

    ##The input is two directories, including ground truth.
    elif len(fileOrDir) == 2:

        fileName_list = load_file_list(fileOrDir[0])
        test_list = get_train_list(load_file_list(fileOrDir[0]), load_file_list(fileOrDir[1]))
        for pair in test_list:
            or_imgY = c_getYdata(pair[0])
            gt_imgY = c_getYdata(pair[1])

            #normalize
            or_imgY = normalize(or_imgY)

            or_imgY = np.resize(or_imgY, (1, or_imgY.shape[0], or_imgY.shape[1],1))
            gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1],1))

            ## act as a placeholder
            or_imgCbCr = 0
            original_ycbcr.append([or_imgY, or_imgCbCr])
            gt_y.append(gt_imgY)
    else:
        print("Invalid Inputs.")
        exit(0)

    return original_ycbcr, gt_y, fileName_list

def test_all_ckpt(modelPath, fileOrDir):
    max = [0, 0]

    tem = [f for f in os.listdir(modelPath) if 'data' in f]
    ckptFiles = sorted([r.split('.data')[0] for r in tem])

    re_psnr = tf.placeholder('float32')
    tf.summary.scalar('re_psnr', re_psnr)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
        shared_model = tf.make_template('shared_model', model)
        output_tensor, weights = shared_model(input_tensor)
        #output_tensor = shared_model(input_tensor)
        output_tensor = tf.clip_by_value(output_tensor, 0., 1.)
        output_tensor = output_tensor * 255

        merged = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter(OUT_DATA_PATH, sess.graph)

        #weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
        saver = tf.train.Saver(weights)
        sess.run(tf.global_variables_initializer())

        original_ycbcr, gt_y, fileName_list = prepare_test_data(fileOrDir)

        for ckpt in ckptFiles:
            epoch = int(ckpt.split('_')[-1].split('.')[0])
            if epoch < 0:
                continue

            saver.restore(sess,os.path.join(modelPath,ckpt))
            total_time, total_psnr = 0, 0
            total_imgs = len(fileName_list)
            for i in range(total_imgs):
                imgY = original_ycbcr[i][0]
                #imgCbCr = original_ycbcr[i][1]
                gtY = gt_y[i] if gt_y else 0

                start_t = time.time()
                out = sess.run(output_tensor, feed_dict={input_tensor: imgY})
                out = np.around(out)
                out = out.astype('int')

                duration_t = time.time() - start_t
                total_time += duration_t

                # save_path = os.path.join(TESTOUT_PATH, str(epoch), os.path.basename(fileName_list[i]))
                # if not os.path.exists(os.path.dirname(save_path)):
                #     os.makedirs(os.path.dirname(save_path))
                # save_test_img(out, imgCbCr, save_path)

                ## gt_y is not empty means 'ground truth' is offered
                if gt_y:
                    p = psnr(out, gtY)
                    total_psnr += p
                    #print("qp52\tepoch:%d\t%s\t%.4f\n"%(epoch,fileName_list[i], p))
                #print("took:%.2fs\t psnr:%.2f name:%s"%(duration_t, p, save_path))
            #print("AVG_DURATION:%.2f\tAVG_PSNR:%.2f"%(total_time/total_imgs, total_psnr/total_imgs))
            avg_psnr = total_psnr/total_imgs
            avg_duration = (total_time/total_imgs)
            if avg_psnr > max[0]:
                max[0] = avg_psnr
                max[1] = epoch

            summary = sess.run(merged, {re_psnr:avg_psnr})
            file_writer.add_summary(summary, epoch)
            tf.logging.warning("AVG_DURATION:%.3f\tAVG_PSNR:%.3f\tepoch:%d"%(avg_duration, avg_psnr, epoch))

        QP = os.path.basename(ORIGINAL_PATH)
        tf.logging.warning("QP:%s\tepoch: %d\tavg_max:%.4f\tdelta:%.4f"%(QP, max[1], max[0], max[0]-NOFILTER[QP]))
    

if __name__ == '__main__':
    test_all_ckpt(MODEL_PATH, [ORIGINAL_PATH, GT_PATH])
