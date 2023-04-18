import tensorflow as tf
import matplotlib.pyplot as plt
import os 
import matplotlib
import numpy as np

matplotlib.use('TkAgg')

train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            channel_shift_range=0.2,
            rotation_range=30,
        )

classes = ["Buses", "Cars", "Motorcycles", "Trucks"]
image_paths = []
for className in classes:
    for file in os.listdir("training_set/" + className)[:15]:
        path = "training_set/" + className + "/" + file
        image_paths.append(path)

print(image_paths)



images = [tf.keras.preprocessing.image.load_img(path, target_size=(224,224)) for path in image_paths]

# Generate augmented images using the ImageDataGenerator
augmented_images = train_data_gen.flow(tf.stack(images), batch_size=len(images))

for i in range(augmented_images[0].shape[0]):
    plt.imshow(augmented_images[0][i])
    plt.axis('off')
    plt.show()