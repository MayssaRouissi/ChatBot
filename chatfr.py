import random
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
import numpy as np
import pickle
import json
import nltk
import tensorflow as tf
from sklearn.model_selection import train_test_split
from nltk.stem import SnowballStemmer
import nltk
# Download the Punkt tokenizer for French
nltk.download('punkt', quiet=True)

stemmer = SnowballStemmer('french')
mots = []
cl = []
doc = []
mots_ignores = ['?', '!']
fichier_json = open('intents_fr.json', encoding='utf-8').read()
intents = json.loads(fichier_json)

entrainement_x = []
entrainement_y = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        mots_tokenize = nltk.word_tokenize(pattern, language='french')
        mots.extend(mots_tokenize)
        doc.append((mots_tokenize, intent['tag']))

        if intent['tag'] not in cl:
            cl.append(intent['tag'])

mots = [stemmer.stem(mot.lower()) for mot in mots if mot not in mots_ignores]
mots = sorted(list(set(mots)))
cl = sorted(list(set(cl)))


print(len(doc), "documents")
print(len(cl), "classes", cl)
print(len(mots), "mots uniques après la racinisation", mots)

pickle.dump(mots, open('mots.pkl', 'wb'))
pickle.dump(cl, open('cl.pkl', 'wb'))

entrainement = []
sortie_vide = [0] * len(cl)

for doc in doc:
    sac_mots = []
    mots_pattern = doc[0]
    mots_pattern = [stemmer.stem(mot.lower()) for mot in mots_pattern]

    for mot in mots:
        sac_mots.append(1) if mot in mots_pattern else sac_mots.append(0)

    sortie_ligne = list(sortie_vide)
    sortie_ligne[cl.index(doc[1])] = 1 

    entrainement.append((sac_mots, sortie_ligne))

random.shuffle(entrainement)

entrainement = np.array(entrainement, dtype=object)

entrainement_x = list(entrainement[:, 0])
entrainement_y = list(entrainement[:, 1])
print("Données d'entraînement créées")

# Créer le modèle - 3 couches. La première couche a 128 neurones, la deuxième couche a 64 neurones,
# et la 3ème couche de sortie contient le nombre de neurones égal au nombre d'intentions pour prédire la sortie avec softmax.

modele = Sequential()

modele.add(Dense(256, input_shape=(len(entrainement_x[0]),), activation='relu'))
modele.add(Dropout(0.5))

modele.add(Dense(128, activation='relu'))
modele.add(Dropout(0.5))


modele.add(Dense(64, activation='relu'))
modele.add(Dropout(0.5))

modele.add(Dense(len(entrainement_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
modele.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

entrainement_x, test_x, entrainement_y, test_y = train_test_split(entrainement_x, entrainement_y, test_size=0.2, random_state=42)

historique = modele.fit(entrainement_x, entrainement_y, epochs=200, batch_size=200, verbose=1)

modele.save('modele_chatbot_francais.keras')
perte, precision = modele.evaluate(test_x, test_y)
print("Perte du test :", perte)
print("Précision du test :", precision)

print("Modèle créé et sauvegardé.")
