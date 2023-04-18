import os
from os import listdir
from PIL import Image
folder_path = "combined_train_set/"

for directory in listdir(folder_path):
    for file in listdir(folder_path + directory):
        img = Image.open(folder_path + directory + os.sep + file)

