# Importiamo le librerie di Keras

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


#Modello CNN :  Input -> Conv(ReLU) -> MaxPooling -> Conv(ReLU) -> MaxPooling -> Flatten -> Dense(128) -> Dense(10)
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(128,128,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#Normalizzazione Dataset
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# andiamo a settare il training set passando la path, la dimensione a cui tutte 
# le immagini devono essere ridimensionate (128,128), la classificazione che sarà multiclasse, e la batch_size a 40
training_set = train_datagen.flow_from_directory('Dataset/TrainingSet',
                                                 target_size=(128, 128),
                                                 batch_size=40,
                                                 class_mode='categorical')

# la stessa cosa del training set la facciamo per il test set
test_set = test_datagen.flow_from_directory('Dataset/TestSet',
                                            target_size=(128, 128),
                                            batch_size=40,
                                            class_mode='categorical')

# impostiamo per la fase di fit il training set, il numero di steps per iterazione(680)
# il numero di iterazioni (10), il test set e il numero di steps per iterazione (170),
# e il parametro shuffle che prende le immagini in maniera casuale
classifier.fit_generator(training_set,
                         steps_per_epoch=680,
                         epochs=10,
                         validation_data=test_set,
                         validation_steps=170,
                         shuffle=True)


# serializza la struttura della rete neurale e i pesi in HDF5 e salva il modello
classifier.save("model.h5")
print("Modello salvato")

