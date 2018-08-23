# Importiamo le librerie di Keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
import PIL

#Inizializzo la CNN
from keras.utils import np_utils

classifier = Sequential()
# Step 1 - aggiungiamo un primo layer convoluzionale di tipo ReLu, specifichiamo il numero di filtri (32) e la loro dimensione 3x3
# specifichiamo inoltre la dimensione delle immagini (64,64,3)
classifier.add(Conv2D(32, (3, 3), input_shape=(128,128,3), activation='relu'))
# Step 2 - Aggiungiamo un pooling layer per ridurre la dimensione delle immagini e diamo come parametri la dimensione della matrice (2x2)
# in modo da perdere pochi pixel e concentrare tutte le features in una regione precisa
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# secondo layer convoluzionale ReLu e successivo pooling layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Step 3 - eseguiamo un'operazione di flattening in modo da ridurre le immagini ad un vettore mono-dimensionale
classifier.add(Flatten())
# Step 4 - aggiungiamo un layer fully-connected impostando il numero di neuroni (128)
classifier.add(Dense(units=128, activation='relu'))
# Step 5 - inizializziamo l'output layer con 1 solo neurone (poichè la scelta è binaria) e utilizzando la 
# funzione di attivazione sigmoide 
classifier.add(Dense(units=10, activation='softmax'))
# Compiliamo la CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Parte 2 - Facciamo delle operazioni preliminari sulle immagini per evitare overfitting, eseguiamo compressione
# ridimensionamento e zoom 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255 #,
                                   #shear_range=0.2,
                                   #zoom_range=0.2,
                                   #horizontal_flip=True
                                   )

test_datagen = ImageDataGenerator(rescale=1. / 255)

# andiamo a settare il training set passando la path, la dimensione a cui tutte 
# le immagini devono essere ridimensionate (64,64) e la classificazione che sarà binaria
training_set = train_datagen.flow_from_directory('Dataset/TrainingSet',
                                                 target_size=(128, 128),
                                                 batch_size=50,
                                                 class_mode='categorical')

# la stessa cosa del training set la facciamo per il test set
test_set = test_datagen.flow_from_directory('Dataset/TestSet',
                                            target_size=(128, 128),
                                            batch_size=50,
                                            class_mode='categorical')
# impostiamo per la fase di fit il training set, il numero di immagini del training (3338)
# il numero di iterazioni (5), il test set e il numero di immagini nel test set (856)




classifier.fit_generator(training_set,
                         steps_per_epoch=480,
                         epochs=5,
                         validation_data=test_set,
                         validation_steps=160,
                         shuffle=True)

#classifier.fit(training_set, y_train, validation_data=(test_set, y_test), epochs=10, batch_size=200)
# serializza la struttura della rete neurale e i pesi in HDF5 e salva il modello
classifier.save("model.h5")
print("Modello salvato")

import numpy as np

# carichiamo il modello
classifier = load_model("model.h5")
print("Caricamento modello")

classifier = load_model("model.h5")
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('Dataset/TestSet',
                                            target_size=(128, 128),
                                            batch_size=64,
                                            class_mode='categorical')

print('Test del modello')

print(classifier.evaluate_generator(test_set,
                              steps=120,
                              #workers=5,
                              #use_multiprocessing=True,
                              verbose=1))

print(classifier.predict_generator(test_set, steps=120, verbose=1))
'''
from keras.preprocessing import image

# carichiamo l'immagine da testare e successivamente facciamo la predizione 
test_image = image.load_img('/Users/lorenzovalente/Desktop/evaluation/vecchi/Vincenzo_old.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
# facciamo la predizione per vedere se l'immagine contiene un bambino o un adulto
result = classifier.predict(test_image)

print("result")
print(result)

# se il rilultato della predizione è uguale a 1 allora è bambino

if result[0][0] == 1:
    prediction = 'young'
    print(prediction)
# altrimenti se è 0 è adulto
else:
    prediction = 'old'
    print(prediction)
'''
