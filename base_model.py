import tensorflow as tf


class BaseModel:
    def __init__(self, input_width, input_height):
        self.input_width = input_width
        self.input_height = input_height
        self.image_input = tf.keras.layers.Input((self.input_width, self.input_height, 3))
        self.input_shape = (self.input_width, self.input_height, 3)

    def get_cnn(self):
        model = tf.keras.models.Sequential()
        # Convolutional Neural Network built from scratch
        model.add(tf.keras.layers.Input(shape=self.input_shape))
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=1, strides=1))

        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
        model.add(tf.keras.layers.Dense(units=512, activation='relu'))

        return model, "CNN"

    def get_vgg16(self):
        model = tf.keras.models.Sequential()

        base_model = tf.keras.applications.VGG16(input_tensor=self.image_input, input_shape=self.input_shape,
                                                 weights='imagenet', include_top=False)
        # Freeze last 3 layers
        for layer in base_model.layers[-3:]:
            layer.trainable = False

        model.add(base_model)

        # Extra layers added at the end of the model
        model.add(tf.keras.layers.GlobalMaxPool2D())
        model.add(tf.keras.layers.Dropout(0.2))

        return model, "VGG16"

    def get_vgg19(self):
        model = tf.keras.models.Sequential()

        base_model = tf.keras.applications.VGG19(input_tensor=self.image_input, input_shape=self.input_shape,
                                                 weights='imagenet', include_top=False)
        # Freeze last 3 layers
        for layer in base_model.layers[-3:]:
            layer.trainable = False

        model.add(base_model)

        # Extra layers added at the end of the model
        model.add(tf.keras.layers.GlobalMaxPool2D())
        model.add(tf.keras.layers.Dropout(0.2))

        return model, "VGG19"

    def get_resnet50v2(self):
        model = tf.keras.models.Sequential()

        base_model = tf.keras.applications.ResNet152V2(input_tensor=self.image_input, input_shape=self.input_shape,
                                                       weights='imagenet', include_top=False)

        model.add(base_model)
        model.add(tf.keras.layers.GlobalMaxPool2D())
        # model.add(tf.keras.layers.Dense(units=1024, activation='relu'))

        # model.add(tf.keras.layers.Dropout(0.2))
        #

        return model, "ResNet152V2"

    def get_inception3(self):
        model = tf.keras.models.Sequential()

        base_model = tf.keras.applications.InceptionV3(input_tensor=self.image_input, input_shape=self.input_shape,
                                                       weights='imagenet', include_top=False, )

        for layer in base_model.layers[:249]:
            layer.trainable = False
        for layer in base_model.layers[249:]:
            layer.trainable = True

        model.add(base_model)

        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.Dense(units=512, activation='relu'))

        return model, "InceptionV3"

    def get_efficient_net_b2(self):
        model = tf.keras.models.Sequential()

        base_model = tf.keras.applications.EfficientNetB2(input_tensor=self.image_input, input_shape=self.input_shape,
                                                          weights='imagenet', include_top=False, )

        model.add(base_model)
        model.add(tf.keras.layers.GlobalMaxPooling2D())

        return model, "EfficientNetB2"

    def get_mobilenet(self):
        model = tf.keras.models.Sequential()

        base_model = tf.keras.applications.MobileNet(input_tensor=self.image_input, input_shape=self.input_shape,
                                                     weights='imagenet', include_top=False)

        model.add(base_model)

        model.add(tf.keras.layers.GlobalMaxPool2D())
        model.add(tf.keras.layers.Dropout(0.1))

        return model, "MobileNet"
