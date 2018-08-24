# Importiamo le librerie di Keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator



#carichiamo il modello salvato
classifier = load_model("model.h5")
print("Caricamento modello")

#ridimensioniamo otteniamo il dataset di test
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('Dataset/TestSet',
                                            target_size=(128, 128),
                                            batch_size=40,
                                            class_mode='categorical')

print('Test del modello')
#valutiamo l'accuratezza del modello
valutazione = classifier.evaluate_generator(test_set,steps=170,verbose=1)
print("L'accuratezza del modello è ",valutazione[1],"Il valore di loss è ", valutazione[0])
#prediction=classifier.predict_generator(test_set, steps=85, verbose=1)
