from os import listdir
from os.path import isfile
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.utils import class_weight
from base_model import BaseModel

img_width = 128
img_height = 128
batch_size = 32


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
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            channel_shift_range=0.3,
            rotation_range=30,
        )

        self.saved_model_path = saved_model_path

        self.training_set = self.train_data_gen.flow_from_directory(
            'dataset/training_set',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
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
            self.model = self.train_model()
        else:
            self.model = self.load_model()

    def train_model(self):
        base_model = BaseModel(img_width, img_height)

        self.model, self.model_name = base_model.get_resnet50v2()

        self.model.add(tf.keras.layers.Dense(units=4, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=0)

        # Save models per epoch
        model_path = "saved_models/Best-VGG19-60-80-85-70.hdf5"
        model_save_callback = tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=False, monitor='loss')

        # Class weights
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(self.training_set.classes),
                                                          y=self.training_set.classes)
        class_weights = {k: v for k, v in enumerate(class_weights)}

        self.model.fit(x=self.training_set, validation_data=self.test_set, epochs=15,
                       steps_per_epoch=13937 // batch_size,
                       validation_steps=2624 // batch_size, class_weight=class_weights,
                       callbacks=[early_stopping, model_save_callback])
        self.model.save('saved_models/model', save_format="h5")

        return self.model

    def load_model(self):
        return tf.keras.models.load_model(self.saved_model_path)

    def predict_single(self, img_name):
        img_path = self.test_images_folder_path + "/" + img_name

        test_image = load_img(img_path, target_size=(img_width, img_height))
        test_image = img_to_array(test_image)
        test_image /= 255.
        test_image = np.expand_dims(test_image, axis=0)
        result = self.model.predict(test_image)[0]
        item_index = np.where(result == max(result))[0][0]

        print("--- ---")
        print("Image name: ", img_name)
        print(result)
        print("Prediction: ", self.classes[item_index])
        for index, prob in enumerate(result):
            print(self.classes[index], ": ", prob)

        return result

    def predict_single_class(self, vehicle_type, img):
        img_path = "dataset/test_set/" + vehicle_type + "/" + img
        test_image = load_img(img_path, target_size=(img_width, img_height))
        test_image = img_to_array(test_image)
        test_image /= 255.
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

    def get_model_name(self, extra=""):
        model_name = self.model_name
        for layer in self.model.layers:
            if layer.name == "dense":
                model_name += "-Dense" + str(layer.units)
            elif layer.name == "dropout":
                model_name += "Dropout" + str(layer.rate).replace(".", "")
        model_name += "-" + str(type(self.model.optimizer).__name__)
        model_name += extra

        return model_name
