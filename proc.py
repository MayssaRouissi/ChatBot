import random
import json
from keras.models import load_model
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
#lematizer is a process of reducding words to their base form 
lemmatizer = WordNetLemmatizer() 




model = load_model('chatbot_model.keras')
# loads the contents of a JSON file into a Python dictionary using the json.loads() function. It assumes that the JSON file is encoded in UTF-8 format.
intents = json.loads(open('intents.json', encoding='utf-8').read())
#The file is opened in binary mode ('rb') to read the pickled data.
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))






def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))



def predict_class(sentence, model):
    # filter out predictions below a threshold 
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    print(float(ints[0]['probability']))
    if float(ints[0]['probability']) > 0.7:
        for i in list_of_intents:
            if (i['tag'] == tag):
                result = random.choice(i['responses'])
                break
            else:
                result = "You must ask the right questions"
    else:
        result = 'I dont understand your question!'
    return result





def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res






# while True:
#     user_input = input("You: ")
#     if user_input.lower() == 'exit':
#         break
#     response = chatbot_response(user_input)
#     print("Chatbot:", response)

