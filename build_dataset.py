from config import lung_config as config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
img_height = config.IMG_HEIGHT
img_width = config.IMG_WIDTH
batch_size = config.BATCH_SIZE

# Create Image Data Generator for Train Set
image_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# Create Image Data Generator for Test/Validation Set
test_data_gen = ImageDataGenerator(rescale=1./255)

train = image_gen.flow_from_directory(
    config.TRAIN_PATH,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=batch_size
)

test = test_data_gen.flow_from_directory(
    config.TEST_PATH,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    shuffle=False,
    # setting shuffle as False just so we can later compare it with predicted values without having indexing problem
    class_mode='binary',
    batch_size=batch_size
)

valid = test_data_gen.flow_from_directory(
    config.VAL_PATH,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=batch_size
)


print(train)

plt.figure(figsize=(12, 12))
for i in range(0, 10):
    plt.subplot(2, 5, i+1)
    for X_batch, Y_batch in train:
        image = X_batch[0]
        dic = {0: 'NORMAL', 1: 'PNEUMONIA'}
        plt.title(dic.get(Y_batch[0]))
        plt.axis('off')
        plt.imshow(np.squeeze(image), cmap='gray', interpolation='nearest')
        break
plt.tight_layout()
plt.show()
