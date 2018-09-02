# Importiamo le librerie di Keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# carichiamo il modello salvato
classifier = load_model("model.h5")
print("Caricamento modello")

# ridimensioniamo otteniamo il dataset di test
test_datagen = ImageDataGenerator(rescale=1. / 255)

# carichiamo il dataset
test_set = test_datagen.flow_from_directory('Dataset/TestSet',
                                            target_size=(128, 128),
                                            batch_size=50,
                                            class_mode='categorical',
                                            shuffle=False)

print('Test del modello')
# valutiamo l'accuratezza del modello
valutazione = classifier.evaluate_generator(test_set, verbose=1)

print("L'accuratezza del modello è ", valutazione[1], "Il valore di loss è ", valutazione[0])

# calcoliamo le predizioni
Y_pred = classifier.predict_generator(test_set, test_set.samples // test_set.batch_size)
y_pred = np.argmax(Y_pred, axis=1)
# calcoliamo la matrice di confusione
cm = confusion_matrix(test_set.classes, y_pred)
# definiamo i nomi delle classi
target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# funzione per disegnare la matrice di confusione


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Matrice di confusione',
                          cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Label effettiva')
    plt.xlabel('Label predetta')
    plt.pause(10)


# stampiamo i risultati
print(classification_report(test_set.classes, y_pred, target_names=target_names))
plot_confusion_matrix(cm, target_names)
