# Approach 1

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

labels = ['exterior', 'interior']
img_size = 224


def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  # convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
                print(img)
    return np.array(data)




def get_model():

    train = get_data(
        '/ajapaik-learning/app/app/algorithms/interior_exterior/input/train')
    test = get_data('/Users/annagrund/PycharmProjects/ajapaik-learning/app/app/algorithms/interior_exterior/input/test')


    # train = get_data('app/storage/input/train')
    # test = get_data('app/storage/input/test')


    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in test:
        x_test.append(feature)
        y_test.append(label)

    # Normalize the data
    x_train = np.array(x_train) / 255
    x_test = np.array(x_test) / 255

    x_train.reshape(-1, img_size, img_size, 1)
    y_train = np.array(y_train)

    x_test.reshape(-1, img_size, img_size, 1)
    y_test = np.array(y_test)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.2,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False

    model = tf.keras.Sequential([base_model,
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.Dropout(0.2),
                                 tf.keras.layers.Dense(2, activation="softmax")
                                 # ValueError: `logits` and `labels` must have the same shape, received ((32, 2) vs (32, 1)).
                                 ])

    model.summary()
    base_learning_rate = 0.00001

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Verify, if that was the issue
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))

    return model


# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs_range = range(2)
#
# plt.figure(figsize=(15, 15))
# plt.subplot(2, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(2, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.savefig('result_interior_vs_exterior.png')
# plt.show()
# #
# # # predictions = model.predict(x_test)
# # # # TODO: try to fix it
# # # predictions = predictions.reshape(1,-1)[0]
# # # print(len(x_test))
# # # print(len(y_test))
# # # print(len(x_train))
# # # print(len(y_train))
# # # print(type(predictions))
# # # print(classification_report(y_test, predictions, target_names = ['Cat (Class 0)', 'Dog (Class 1)']))
