# Importiamo le librerie di Keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import optimizers
from keras.models import model_from_json
import PIL
import numpy
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


classifier = load_model("model.h5")
test_datagen = ImageDataGenerator()
test_set = test_datagen.flow_from_directory('Dataset/TestSet',
                                            target_size=(128, 128),
                                            batch_size=64,
                                            class_mode='categorical')

print('Test del modello')

print(classifier.evaluate_generator(test_set,
                              steps=160,
                              #workers=5,
                              #use_multiprocessing=True,
                              verbose=1))

print(classifier.predict_generator(test_set, steps=160, verbose=1))