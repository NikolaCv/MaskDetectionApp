import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
import json
from math import floor, ceil
import cv2
import os
from datetime import datetime


def printt(matrix):
    for x in matrix:
        print(x)


def resize_image(img, new_size):
    height = np.size(img, axis=0)
    width = np.size(img, axis=1)

    if height > width:
        new_width = floor(new_size * width / height)
        img = cv2.resize(img, dsize=(new_width, new_size), interpolation=cv2.INTER_CUBIC)
        img = cv2.copyMakeBorder(img, top=0, bottom=0, left=floor((new_size - new_width) / 2),
                                 right=ceil((new_size - new_width) / 2),
                                 borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        new_height = floor(new_size * height / width)
        img = cv2.resize(img, dsize=(new_size, new_height), interpolation=cv2.INTER_CUBIC)
        img = cv2.copyMakeBorder(img, top=floor((new_size - new_height) / 2), bottom=ceil((new_size - new_height) / 2),
                                 left=0, right=0,
                                 borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return np.reshape(img, (1, new_size, new_size, 3))


def generator_from_folder(folder, new_size, loop):
    while 1:
        for ind, filename in enumerate(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, filename))

            if img is not None:
                info = filename.split('_')
                one_hot = [0, 0, 0, 0]
                one_hot[int(info[1]) - 1] = 1

                yield resize_image(img, new_size), np.reshape(one_hot, (1, 4, 1))

        if not loop:
            break

if __name__ == '__main__':
    time = '(' + datetime.now().strftime('%d.%m.%Y. - %H:%M:%S')
    new_img_size = 300
    print(time)
    # creating nn model
    ResNet = ResNet50(
        include_top=None, weights='imagenet', input_tensor=None, input_shape=(new_img_size, new_img_size, 3),
        pooling=None, classes=5)

    # ResNet.summary()

    # adding flat layer at the end, for 4 outputs
    model = keras.models.Sequential()
    model.add(ResNet)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4, activation='softmax', kernel_regularizer=keras.regularizers.l1(1e-2)))

    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    ###############################################

    time = '(' + datetime.now().strftime('%d.%m.%Y. - %H:%M:%S')

    train_data = 'rain_images'
    batch_size = 32
    epochs = 50

    # 2 corrupted images
    N = len(os.listdir(train_data)) - 1

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='loss', patience=10),
        keras.callbacks.ModelCheckpoint('Mask_classification', monitor='val_accuracy', mode='max', verbose=1,
                                        save_best_only=True)
    ]

    history = model.fit(generator_from_folder(train_data, new_img_size, True), epochs=epochs, steps_per_epoch=N,
                        batch_size=batch_size, verbose=1, callbacks=callbacks)

    print(history.history)

    with open('history.txt', 'w') as f:
        f.write(json.dumps(history.history))


    time += ' – ' +  datetime.now().strftime('%d%m%Y - %H:%M:%S') + ')'

    model.save('Mask_classification ' + time)

    a = input('Press any key to continue...')
