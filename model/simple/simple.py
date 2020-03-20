# coding: utf-8
import os
from typing import Union

from tensorflow.keras import layers, Input, models


def Simple_model(input_shape, classes):
    if input_shape is None:
        raise ValueError('input_shape 不能为空')

    img_input = Input(shape=input_shape, name='img_input')

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(classes))

    return model
