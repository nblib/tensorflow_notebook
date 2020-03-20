# coding: utf-8
import os
from typing import Union

from tensorflow.keras import layers, Input, Model

import tensorflow as tf

tf.keras.applications.resnet

def VGG16(input_shape: Union[list, tuple], include_top=True, classes=10, weights_path=''):
    """
    构建一个VGG16网络, VGG16,表示包含参数的层数有16层(包含全连接层和输出层)
    :param input_shape: 输入shape: 比如： 【224， 224，1】
    :param include_top: 是否包含最后的分类层
    :param classes: 如果包含最后的分类层，设置分类数量
    :param weights_path: 加载训练好的模型路径
    :return: VGG16模型
    :raise: 如果输入shape或weights加载错误(不存在不报错)
    """
    if input_shape is None:
        raise ValueError('input_shape 不能为空')

    img_input = Input(shape=input_shape, name='img_input')

    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # output
    if include_top:
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(256, activation='relu', use_bias=True, name='fc1')(x)
        x = layers.Dense(256, activation='relu', use_bias=True, name='fc2')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    # build model
    model = Model(img_input, x)

    # load trained

    if os.path.exists(weights_path):
        model.load_weights(weights_path)

    return model
