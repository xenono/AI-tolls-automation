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
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
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

        model.add(tf.keras.layers.Dense(units=512, activation='relu'))

        return model, "CNN"

    def get_vgg16(self):
        model = tf.keras.models.Sequential()

        # Transfer learning model
        base_model = tf.keras.applications.VGG16(input_tensor=self.image_input, input_shape=self.input_shape,
                                                 weights='imagenet', include_top=False)
        # Freeze last 3 layers
        for layer in base_model.layers[:-3]:
            layer.trainable = False

        model.add(base_model)

        # Extra layers added at the end of the model
        model.add(tf.keras.layers.GlobalMaxPool2D())
        model.add(tf.keras.layers.Dense(units=512, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))

        return model, "VGG16"

    def get_vgg19(self):
        model = tf.keras.models.Sequential()

        # Transfer learning model
        base_model = tf.keras.applications.VGG19(input_tensor=self.image_input, input_shape=self.input_shape,
                                                 weights='imagenet', include_top=False)
        # Freeze last 3 layers
        for layer in base_model.layers[:-3]:
            layer.trainable = False

        model.add(base_model)

        # Extra layers added at the end of the model
        model.add(tf.keras.layers.GlobalMaxPool2D())
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.Dropout(0.2))

        return model, "VGG19"

    def get_resnet50v2(self):
        model = tf.keras.models.Sequential()

        base_model = tf.keras.applications.ResNet101V2(input_tensor=self.image_input, input_shape=self.input_shape,
                                                       weights='imagenet', include_top=False, pooling='max')

        for layer in base_model.layers[16:]:
            layer.trainable = False

        for layer in base_model.layers[:-15]:
            layer.trainable = False

        model.add(base_model)

        # model.add(tf.keras.layers.GlobalMaxPool2D())
        # model.add(tf.keras.layers.Dropout(0.2))

        return model, "ResNet50V2"