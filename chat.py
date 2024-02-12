import random
#SGD is a popular optimization algorithm commonly used for training neural networks.
from keras.optimizers import SGD
#layer types 
from keras.layers import Dense, Activation, Dropout
#Sequential models 
from keras.models import Sequential
import numpy as np
#  pickle=used for serialization and deserialization of Python objects
import pickle
import json
import nltk

import tensorflow as tf
from sklearn.model_selection import train_test_split
#tokenization capabilities, which can be useful for splitting text into individual words or sentences
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()


words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(data_file)




train_inputs = []
train_labels = []



for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        words.extend(w)

        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower())
         for w in words if w not in ignore_words]
words = sorted(list(set(words)))
#converts the words list to a set to remove duplicates and then converts it back to a list
classes = sorted(list(set(classes)))

print(len(documents), "documents")

print(len(classes), "classes", classes)

print(len(words), "unique lemmatized words", words)


# sertlization le contenu de words into word.pkl by wb (write b)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# initializing training data

#***********************
training = []
#creates a list with the same number of elements as classes, where each element is initialized to 0
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(
        word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append((bag, output_row))

random.shuffle(training)


#dtype=object =accommodate elements of any data type
training = np.array(training, dtype=object)

# create train and test lists. X - patterns, Y - intents

train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

#************************************************


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax


# Create a sequential model
model = Sequential()

# Add layers to the model
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model with the optimizer and loss function
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Fit the model to the training data
#hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=200, verbose=1)
history = model.fit(train_x, train_y, epochs=200, batch_size=200, verbose=1)

# Save the model
model.save('chatbot_model.keras')
loss, accuracy = model.evaluate(test_x, test_y)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

print("Model created and saved.")
 







 