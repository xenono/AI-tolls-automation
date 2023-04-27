from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.transform import resize
from os import sep, listdir, mkdir, remove
from PIL import Image


class DataPreprocessing:
    def __init__(self, dataset_folder, training_set_folder_name, test_set_folder_name, classes, img_width, img_height):
        self.dataset_folder = dataset_folder
        self.classes = classes
        self.training_set_folder_name = training_set_folder_name
        self.test_set_folder_name = test_set_folder_name
        self.img_width = img_width
        self.img_height = img_height

    def scale_dataset(self):
        for dataset in [self.training_set_folder_name, self.test_set_folder_name]:
            scaled_dataset = self.dataset_folder + sep + dataset + "_scaled"
            mkdir(scaled_dataset)
            for folder in listdir(self.dataset_folder + sep + dataset):
                self.__scale_images_in_folder(self.dataset_folder + sep + folder, scaled_dataset + sep + folder)

    def find_corrupted_images_in_folder(self, folder, remove_image=True):
        for file in listdir(folder):
            try:
                img = Image.open(folder + sep + file)
            except IOError:
                print("Found a corrupted image")
                if remove_image:
                    print("Remove image flag detected.")
                    print("Removing the image.")
                    remove(folder + sep + file)

    def __scale_images_in_folder(self, folder_path, new_folder_path):
        mkdir(new_folder_path)
        file_counter = 0
        for file in listdir(folder_path):
            try:
                img = imread(folder_path + sep + file)

                res = resize(img, (self.img_width, self.img_height), anti_aliasing=True)
                file = file.replace(" ", "-")
                print(file)
                imsave(new_folder_path + sep + file, img_as_float(res))
                file_counter += 0
                print(str(file_counter) + ' images processed')
            except IOError:
                print("Cannot read the file, image will not be scaled")
                print("Continue...")
                continue
