from math import ceil, floor
import os
import numpy as np
import psycopg2
import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.utils import shuffle
from PIL import Image
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

images = []
tags = []
tags2 = []
clazz = 0
classes_tags = os.listdir('./training_data')


def read_data():
    return extract_images(open('data/emnist-bymerge-train-images-idx3-ubyte.gz', 'rb')), \
           extract_labels(open('data/emnist-bymerge-train-labels-idx1-ubyte.gz', 'rb'))


train_x, train_y = read_data()
classes_names = np.unique(train_y)


def augment_data(data, data_labels, use_random_shear=True, use_random_shift=True):
    augmented_image = []
    augmented_image_labels = []

    for index in range(0, data.shape[0]):
        # original image:
        augmented_image.append(data[index])
        augmented_image_labels.append(data_labels[index])

        if use_random_shear:
            augmented_image.append(
                tf.contrib.keras.preprocessing.image.random_shear(data[index], 0.5, row_axis=0, col_axis=0,
                                                                  channel_axis=0))
            augmented_image_labels.append(data_labels[index])

        if use_random_shift:
            augmented_image.append(
                tf.contrib.keras.preprocessing.image.random_shift(data[index], 0.5, 0.5, row_axis=0, col_axis=0,
                                                                  channel_axis=0))
            augmented_image_labels.append(data_labels[index])

    return np.array(augmented_image), np.array(augmented_image_labels)


def save(img, tag, destination_path):
    img = array_to_img(img)
    folder = classes_tags[tag]

    path = destination_path + "\\" + str(folder)

    if not os.path.exists(path):
        os.makedirs(path)
    index_image = 1+ max([int(f.split(".")[0]) for f in os.listdir(path)], default=0)
    img.save(path + "\\" + str(index_image) + ".png")


def generate_augmented_data_from_folder(source_path, destination_path, img_rows, img_cols):
    for root, dirs, files in os.walk(source_path):
        global clazz
        global tags
        global images
        try:
            for f in files:
                images.append(img_to_array(load_img(root + '/' + f, grayscale=True, target_size=(img_rows, img_cols))))
                tags.append(clazz)
                if len(tags) % 2534 == 0:
                    break

            tags = [t - 1 for t in tags]
            images = np.array(images)
            tags = np.array(tags)

            augum_img, augum_tags = augment_data(images, tags, use_random_shear=True, use_random_shift=True)

            for index in range(len(augum_img)):
                save(augum_img[index], augum_tags[index], destination_path)
            print(root)
            clazz += 1
            images=[]
            tags = []
            augum_img=[]
            augum_tags=[]
        except IndexError:
            pass

			
if __name__ == '__main__':
    generate_augmented_data_from_folder('./training_data', './training_data_augm', 28, 28)
