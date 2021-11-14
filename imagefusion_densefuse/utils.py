# Utility

import numpy as np

from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from scipy.misc import imread, imsave, imresize
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
from PIL import Image
from functools import reduce

def list_images(directory):
    images = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
    return images


# read images
def get_image(path, height=256, width=256, set_mode='L'):
    image = imread(path, mode=set_mode)
    if height is not None and width is not None:
        image = imresize(image, [height, width], interp='nearest')
    return image


def get_train_images(paths, resize_len=512, crop_height=256, crop_width=256, flag=True):
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        image = get_image(path, height=crop_height, width=crop_width, set_mode='L')

        if flag:
            image = np.stack(image, axis=0)
            image = np.stack((image, image, image), axis=-1)
        else:
            image = np.stack(image, axis=0)
            image = image.reshape([crop_height, crop_width, 1])
        images.append(image)
    images = np.stack(images, axis=-1)
    return images


def get_train_images_rgb(paths, crop_height=256, crop_width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        image = get_image(path, height=crop_height, width=crop_width, set_mode='RGB')
        image = np.stack(image, axis=0)
        images.append(image)
    images = np.stack(images, axis=-1)
    return images


def get_test_image_rgb(path, resize_len=512, crop_height=256, crop_width=256, flag = True):
    # image = imread(path, mode='L')
    image = imread(path, mode='RGB')
    return image


def get_images_test(path, mod_type='L', height=None, width=None):

    image = imread(path, mode=mod_type)
    if height is not None and width is not None:
        image = imresize(image, [height, width], interp='nearest')

    if mod_type=='L':
        d = image.shape
        image = np.reshape(image, [d[0], d[1], 1])

    return image


def get_images(paths, height=None, width=None):
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        image = imread(path, mode='RGB')

        if height is not None and width is not None:
            image = imresize(image, [height, width], interp='nearest')

        images.append(image)

    images = np.stack(images, axis=0)
    print('images shape gen:', images.shape)
    return images


def save_images(paths, datas, save_path, prefix=None, suffix=None):
    if isinstance(paths, str):
        paths = [paths]

    t1 = len(paths)
    t2 = len(datas)
    assert(len(paths) == len(datas))

    if not exists(save_path):
        mkdir(save_path)

    if prefix is None:
        prefix = ''
    if suffix is None:
        suffix = ''

    for i, path in enumerate(paths):
        data = datas[i]
        # print('data ==>>\n', data)
        if data.shape[2] == 1:
            data = data.reshape([data.shape[0], data.shape[1]])
        # print('data reshape==>>\n', data)

        name, ext = splitext(path)
        name = name.split(sep)[-1]
        
        path = join(save_path, prefix + suffix + ext)
        print('data path==>>', path)


        # new_im = Image.fromarray(data)
        # new_im.show()

        imsave(path, data)

def get_l2_norm_loss(diffs):
    shape = diffs.get_shape().as_list()
    size = reduce(lambda x, y: x * y, shape) ** 2
    sum_of_squared_diffs = tf.reduce_sum(tf.square(diffs))
    return sum_of_squared_diffs / size