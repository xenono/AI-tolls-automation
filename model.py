import sys  # to access the system
import os
from os import listdir, system
from os.path import isfile, join, isdir
import logging
import tensorflow as tf
import numpy as np
from keras import utils
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import SGD
from PySide6 import QtCore, QtWidgets, QtGui
from sklearn.utils import class_weight

img_width = 64
img_height = 64
batch_size = 64


class Model:
    def __init__(self, saved_model_path):
        # Allow memory growth for the GPU
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.test_images_folder_path = "dataset/manual_test"
        self.model_name = "CNN"
        # Preprocessing dataset
        self.train_data_gen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.4,
            zoom_range=0.4,
            horizontal_flip=True,
            vertical_flip=True,
            channel_shift_range=0.2,
            rotation_range=30,
        )

        self.saved_model_path = saved_model_path

        self.training_set = self.train_data_gen.flow_from_directory(
            'dataset/training_set',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical'
        )

        self.test_set_data_gen = ImageDataGenerator(rescale=1. / 255)
        self.test_set = self.test_set_data_gen.flow_from_directory(
            'dataset/test_set',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical'
        )
        self.classes = {}
        for key, value in self.training_set.class_indices.items():
            self.classes[value] = (key[:-1] if key != "buses" else key[:-2]).capitalize()
        if not isfile(self.saved_model_path):
            self.model = self.construct_cnn()
        else:
            self.model = self.load_model()

        # self.model.evaluate(self.test_set)
        print(self.training_set.class_indices)
        self.get_per_class_accuracy(["Buses", "Cars", "Motorcycles", "Trucks"])

    def construct_cnn(self):
        image_input = tf.keras.layers.Input((img_width, img_height, 3))
        model = tf.keras.models.Sequential()
        # Transfer learning model
        base_model = tf.keras.applications.VGG16(input_tensor=image_input, input_shape=(img_width, img_height, 3),
                                                 include_top=False, weights="imagenet")

        model_folder_name = base_model.name.upper()

        for layer in base_model.layers[:-2]:
            layer.trainable = False

        model.add(base_model)
        model.add(tf.keras.layers.GlobalMaxPool2D())
        # model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))

        for layer in model.layers[1:]:
            model_folder_name += "-" + layer.name
            if layer.name == "dense":
                model_folder_name += str(layer.units)
            if layer.name == "dropout":
                model_folder_name += str(layer.rate).replace(".","")

        # End of transfer learning
        # Self-made model
        # model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
        # model.add(tf.keras.layers.MaxPool2D(pool_size=1, strides=1))
        #
        # model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        # model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        #
        # model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        # model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        #
        # model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        # model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        #
        # model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        # model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        # model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        # model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu'))
        # model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.GlobalMaxPool2D())
        #
        # model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        # model.add(tf.keras.layers.Dropout(0.2))
        # model.add(tf.keras.layers.Dense(units=512, activation='relu'))
        # model.add(tf.keras.layers.Dense(units=256, activation='relu'))
        # model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        # End of self-made mode

        model.add(tf.keras.layers.Dense(units=4, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model_folder_name += "-" + str(type(model.optimizer).__name__)
        model_folder_name += "-Unfreeze-2-last-layers"
        # Stop training when a monitored quantity has stopped improving
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=0)
        # Save the best model
        file_path = "saved_models/" + model_folder_name + "/Epoch{epoch:02d}-L{loss:.2f}-A{accuracy:.2f}-VL{val_loss:.2f}-VA{val_accuracy:.2f}.hdf5"
        best_model = tf.keras.callbacks.ModelCheckpoint(file_path, save_best_only=False, monitor='loss')

        # Class weights
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(self.training_set.classes),
                                                          y=self.training_set.classes)
        class_weights = {k: v for k, v in enumerate(class_weights)}

        model.fit(x=self.training_set, validation_data=self.test_set, epochs=40, steps_per_epoch=13937 // batch_size,
                  validation_steps=2624 // batch_size, class_weight=class_weights,
                  callbacks=[early_stopping, best_model])
        model.save('saved_models/model', save_format="h5")

        return model

    def load_model(self):
        return tf.keras.models.load_model(self.saved_model_path)

    def predict_single(self, img_name):
        img_path = self.test_images_folder_path + "/" + img_name

        test_image = load_img(img_path, target_size=(img_width, img_height))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        predictions = (tf.nn.softmax(self.model.predict(test_image)[0])).numpy()
        result = self.model.predict(test_image)[0]
        item_index = np.where(result == max(result))[0][0]

        print("--- ---")
        print("Image name: ", img_name)
        print(predictions)
        print("Prediction: ", self.classes[item_index])
        for index, prob in enumerate(result):
            print(self.classes[index], ": ", prob)

        return predictions

    def predict_single_class(self, vehicle_type, img):
        img_path = "dataset/test_set/" + vehicle_type + "/" + img
        test_image = load_img(img_path, target_size=(img_width, img_height))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = self.model.predict(test_image, verbose=0)[0]

        return np.where(result == max(result))[0][0]

    def get_per_class_accuracy(self, classes):
        print("\n")
        score_per_class = {}
        score = 0
        img_count = 0
        for index, type_directory in enumerate(classes):
            print(type_directory + " start")
            for img in listdir("dataset/test_set/" + type_directory + "/"):
                print(self.predict_single_class(type_directory, img), end=" ")
                if self.predict_single_class(type_directory, img) == index:
                    score += 1
                img_count += 1
                if img_count % 115 == 0:
                    print()
            score_per_class[type_directory] = score * 100 / img_count
            score = 0
            img_count = 0
            print("\n" + type_directory, " done")

        print("\n--- Per class accuracy -- \n")
        for key, value in score_per_class.items():
            print(key, ": ", value, "%")

    def predict_multiple(self):
        test_images = listdir(self.test_images_folder_path)

        for img in test_images:
            self.predict_single(img)
