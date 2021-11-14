# Train the DenseFuse Net

from __future__ import print_function

import scipy.io as scio
import numpy as np
import tensorflow as tf
import os

from ssim_loss_function import SSIM_LOSS
from densefuse_net import DenseFuseNet
from utils import get_train_images, get_train_images_rgb

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

HEIGHT = 256
WIDTH = 256
CHANNELS = 1 # gray scale, default

LEARNING_RATE = 1e-4
EPSILON = 1e-5


def train_recons(original_imgs_path, validatioin_imgs_path, save_path, model_pre_path, ssim_weight, EPOCHES_set, BATCH_SIZE, IS_Validation, debug=False, logging_period=1):
    if debug:
        from datetime import datetime
        start_time = datetime.now()
    EPOCHS = EPOCHES_set
    print("EPOCHES   : ", EPOCHS)
    print("BATCH_SIZE: ", BATCH_SIZE)

    num_val = len(validatioin_imgs_path)
    num_imgs = len(original_imgs_path)
    # num_imgs = 100
    original_imgs_path = original_imgs_path[:num_imgs]
    mod = num_imgs % BATCH_SIZE

    print('Train images number %d.\n' % num_imgs)
    print('Train images samples %s.\n' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]

    # get the traing image shape
    INPUT_SHAPE_OR = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        original = tf.placeholder(tf.float32, shape=INPUT_SHAPE_OR, name='original')
        source = original

        print('source  :', source.shape)
        print('original:', original.shape)

        # create the deepfuse net (encoder and decoder)
        dfn = DenseFuseNet(model_pre_path)
        generated_img = dfn.transform_recons(source)
        print('generate:', generated_img.shape)

        ssim_loss_value = SSIM_LOSS(original, generated_img)
        pixel_loss = tf.reduce_sum(tf.square(original - generated_img))
        pixel_loss = pixel_loss/(BATCH_SIZE*HEIGHT*WIDTH)
        ssim_loss = 1 - ssim_loss_value

        loss = ssim_weight*ssim_loss + pixel_loss
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        sess.run(tf.global_variables_initializer())

        # saver = tf.train.Saver()
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        # ** Start Training **
        step = 0
        count_loss = 0
        n_batches = int(len(original_imgs_path) // BATCH_SIZE)
        val_batches = int(len(validatioin_imgs_path) // BATCH_SIZE)

        if debug:
            elapsed_time = datetime.now() - start_time
            print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
            print('Now begin to train the model...\n')
            start_time = datetime.now()

        Loss_all = [i for i in range(EPOCHS * n_batches)]
        Loss_ssim = [i for i in range(EPOCHS * n_batches)]
        Loss_pixel = [i for i in range(EPOCHS * n_batches)]
        Val_ssim_data = [i for i in range(EPOCHS * n_batches)]
        Val_pixel_data = [i for i in range(EPOCHS * n_batches)]
        for epoch in range(EPOCHS):

            np.random.shuffle(original_imgs_path)

            for batch in range(n_batches):
                # retrive a batch of content and style images

                original_path = original_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
                ### read gray scale images
                original_batch = get_train_images(original_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                ### read RGB images
                # original_batch = get_train_images_rgb(original_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                original_batch = original_batch.transpose((3, 0, 1, 2))

                # print('original_batch shape final:', original_batch.shape)

                # run the training step
                sess.run(train_op, feed_dict={original: original_batch})
                step += 1
                if debug:
                    is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                    if is_last_step or step % logging_period == 0:
                        elapsed_time = datetime.now() - start_time
                        _ssim_loss, _loss, _p_loss = sess.run([ssim_loss, loss, pixel_loss], feed_dict={original: original_batch})
                        Loss_all[count_loss] = _loss
                        Loss_ssim[count_loss] = _ssim_loss
                        Loss_pixel[count_loss] = _p_loss
                        print('epoch: %d/%d, step: %d,  total loss: %s, elapsed time: %s' % (epoch, EPOCHS, step, _loss, elapsed_time))
                        print('p_loss: %s, ssim_loss: %s ,w_ssim_loss: %s ' % (_p_loss, _ssim_loss, ssim_weight * _ssim_loss))

                        # IS_Validation = True;
                        # Calculating the accuracy rate for 1000 images, every 100 steps
                        if IS_Validation:
                            val_ssim_acc = 0
                            val_pixel_acc = 0
                            np.random.shuffle(validatioin_imgs_path)
                            val_start_time = datetime.now()
                            for v in range(val_batches):
                                val_original_path = validatioin_imgs_path[v * BATCH_SIZE:(v * BATCH_SIZE + BATCH_SIZE)]
                                val_original_batch = get_train_images(val_original_path, crop_height=HEIGHT, crop_width=WIDTH,flag=False)
                                val_original_batch = val_original_batch.reshape([BATCH_SIZE, 256, 256, 1])
                                val_ssim, val_pixel = sess.run([ssim_loss, pixel_loss], feed_dict={original: val_original_batch})
                                val_ssim_acc = val_ssim_acc + (1 - val_ssim)
                                val_pixel_acc = val_pixel_acc + val_pixel
                            Val_ssim_data[count_loss] = val_ssim_acc/val_batches
                            Val_pixel_data[count_loss] = val_pixel_acc / val_batches
                            val_es_time = datetime.now() - val_start_time
                            print('validation value, SSIM: %s, Pixel: %s, elapsed time: %s' % (val_ssim_acc/val_batches, val_pixel_acc / val_batches, val_es_time))
                            print('------------------------------------------------------------------------------')
                        count_loss += 1


        # ** Done Training & Save the model **
        saver.save(sess, save_path)

        loss_data = Loss_all[:count_loss]
        os.mknod('./models/loss/DeepDenseLossData'+str(ssim_weight)+'.mat')
        scio.savemat('./models/loss/DeepDenseLossData'+str(ssim_weight)+'.mat',{'loss':loss_data})

        loss_ssim_data = Loss_ssim[:count_loss]
        os.mknod('./models/loss/DeepDenseLossSSIMData'+str(ssim_weight)+'.mat')
        scio.savemat('./models/loss/DeepDenseLossSSIMData'+str(ssim_weight)+'.mat', {'loss_ssim': loss_ssim_data})

        loss_pixel_data = Loss_pixel[:count_loss]
        os.mknod('./models/loss/DeepDenseLossPixelData'+str(ssim_weight)+'.mat')
        scio.savemat('./models/loss/DeepDenseLossPixelData'+str(ssim_weight)+'.mat', {'loss_pixel': loss_pixel_data})

        # IS_Validation = True;
        if IS_Validation:
            validation_ssim_data = Val_ssim_data[:count_loss]
            scio.savemat('./models/val/Validation_ssim_Data.mat' + str(ssim_weight) + '', {'val_ssim': validation_ssim_data})
            validation_pixel_data = Val_pixel_data[:count_loss]
            scio.savemat('./models/val/Validation_pixel_Data.mat' + str(ssim_weight) + '', {'val_pixel': validation_pixel_data})


        if debug:
            elapsed_time = datetime.now() - start_time
            print('Done training! Elapsed time: %s' % elapsed_time)
            print('Model is saved to: %s' % save_path)

