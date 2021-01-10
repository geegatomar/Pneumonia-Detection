import config.lung_config as config
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create Image Data Generator for Train Set
image_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
# Create Image Data Generator for Test/Validation Set
test_data_gen = ImageDataGenerator(rescale=1./255)
