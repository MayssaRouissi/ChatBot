import random
import json
from keras.models import load_model
import numpy as np
import pickle
import nltk
from nltk.stem import SnowballStemmer


stemmer = SnowballStemmer('french')

model = load_model('modele_chatbot_francais.keras')
ints = json.loads(open('intents_fr.json', encoding='utf-8').read())
mots = pickle.load(open('mots.pkl', 'rb'))
cl = pickle.load(open('cl.pkl', 'rb'))


def nettoyer_phrase(phrase):
    mots_phrase = nltk.word_tokenize(phrase, language='french')
    mots_phrase = [stemmer.stem(mot.lower()) for mot in mots_phrase]
    return mots_phrase


def sac_de_mots(phrase, mots, montrer_details=True):
    mots_phrase = nettoyer_phrase(phrase)
    sac = [0] * len(mots)
    for s in mots_phrase:
        for i, w in enumerate(mots):
            if w == s:
                sac[i] = 1
                if montrer_details:
                    print("trouvé dans le sac : %s" % w)
    return np.array(sac)


def prevoir_classe(phrase, model):
    p = sac_de_mots(phrase, mots, montrer_details=False)
    res = model.predict(np.array([p]))[0]
    SEUIL_ERREUR = 0.25
    res = [[i, r] for i, r in enumerate(res) if r > SEUIL_ERREUR]
    res.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in res:
        return_list.append({"intent": cl[r[0]], "probability": str(r[1])})
    return return_list


def obtenir_reponse(intents_list, intentsfr_json):
    tags = intents_list[0]['intent']
    list_of_intents = intentsfr_json['intents']
    print(float(intents_list[0]['probability']))
    if float(intents_list[0]['probability']) > 0.7:
        for i in list_of_intents:
            if (i['tag'] == tags):
                result = random.choice(i['responses'])
                break
            else:
                result = "Vous devez poser les bonnes questions"
    else:
        result = "Je ne comprends pas votre question ! Si vou voulez envoyer un mail pour plus d'info"
    return result


def reponse_chatbot(message):
    intents_list = prevoir_classe(message, model)
    response = obtenir_reponse(intents_list, ints)
    return response


# while True:
#     user_input = input("Vous: ")
#     if user_input.lower() == 'exit':
#         break
#     response = obtenir_reponse(prevoir_classe(user_input, model), ints)
#     print("Chatbot:", response)
