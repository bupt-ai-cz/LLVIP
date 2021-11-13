# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
import cv2

#reader = tf.train.NewCheckpointReader("./checkpoint/CGAN_120/CGAN.model-9")


def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    #flatten=True 以灰度图的形式读取 
    #return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    image=scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    return scipy.misc.imresize(image,0.5)
  else:
    #return scipy.misc.imread(path, mode='YCbCr').astype(np.float)
    image=scipy.misc.imread(path, mode='YCbCr').astype(np.float)
    return scipy.misc.imresize(image,0.5)

def imsave(image, path):
  return scipy.misc.imsave(path, image)
  
  
def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def fusion_model(img):
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1'):
            weights=tf.get_variable("w1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/w1')))
            bias=tf.get_variable("b1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/b1')))
            conv1_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_ir = lrelu(conv1_ir)
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/w2')))
            bias=tf.get_variable("b2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/b2')))
            conv2_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_ir, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_ir = lrelu(conv2_ir)
        with tf.variable_scope('layer3'):
            weights=tf.get_variable("w3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/w3')))
            bias=tf.get_variable("b3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/b3')))
            conv3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_ir, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_ir = lrelu(conv3_ir)
        with tf.variable_scope('layer4'):
            weights=tf.get_variable("w4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/w4')))
            bias=tf.get_variable("b4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/b4')))
            conv4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_ir, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_ir = lrelu(conv4_ir)
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5')))
            bias=tf.get_variable("b5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/b5')))
            conv5_ir= tf.nn.conv2d(conv4_ir, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv5_ir=tf.nn.tanh(conv5_ir)
    return conv5_ir
    




def input_setup(index):
    padding=6
    sub_ir_sequence = []
    sub_vi_sequence = []
    input_ir=(imread(data_ir[index])-127.5)/127.5
    input_ir=np.lib.pad(input_ir,((padding,padding),(padding,padding)),'edge')
    w,h=input_ir.shape
    input_ir=input_ir.reshape([w,h,1])
    input_vi=(imread(data_vi[index])-127.5)/127.5
    input_vi=np.lib.pad(input_vi,((padding,padding),(padding,padding)),'edge')
    w,h=input_vi.shape
    input_vi=input_vi.reshape([w,h,1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir= np.asarray(sub_ir_sequence)
    train_data_vi= np.asarray(sub_vi_sequence)
    return train_data_ir,train_data_vi


num_epoch=9
while(num_epoch==9):

    reader = tf.train.NewCheckpointReader('./checkpoint_20/CGAN_120/CGAN.model-'+ str(num_epoch))

    with tf.name_scope('IR_input'):
        #红外图像patch
        images_ir = tf.placeholder(tf.float32, [1,None,None,None], name='images_ir') 
    with tf.name_scope('VI_input'):
        #可见光图像patch
        images_vi = tf.placeholder(tf.float32, [1,None,None,None], name='images_vi') 
        #self.labels_vi_gradient=gradient(self.labels_vi)
    #将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
    with tf.name_scope('input'):
        #resize_ir=tf.image.resize_images(images_ir, (512, 512), method=2)
        input_image=tf.concat([images_ir,images_vi],axis=-1)
    with tf.name_scope('fusion'):
        fusion_image=fusion_model(input_image)


    with tf.Session() as sess: 
        init_op=tf.global_variables_initializer() 
        sess.run(init_op)
        data_ir=prepare_data('Test_LLVIP_ir')
        data_vi=prepare_data('Test_LLVIP_vi')
        for i in range(len(data_ir)):
            start=time.time()
            train_data_ir,train_data_vi=input_setup(i)
            result =sess.run(fusion_image,feed_dict={images_ir: train_data_ir,images_vi: train_data_vi})
            result=result*127.5+127.5
            result = result.squeeze()
            image_path = os.path.join(os.getcwd(), 'result','epoch'+str(num_epoch))
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            if i<=9:
                image_path = os.path.join(image_path,'F9_0'+str(i)+".jpg")
            else:
                image_path = os.path.join(image_path,'F9_'+str(i)+".jpg")
            end=time.time()
            # print(out.shape)
            imsave(result, image_path)
            print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
    tf.reset_default_graph() 
    num_epoch=num_epoch+1
