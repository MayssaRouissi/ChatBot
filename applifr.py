import io
from flask import Flask, render_template, jsonify, request

import proc
import procfr
from flask_cors import CORS
from flask_restful import Api

app = Flask(__name__)
CORS(app)


app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())

    #  return render_template('aaaaaa.html', **locals())

# def detect_language(message):
#     # Vérifier si le message contient plus de voyelles que de consonnes
#     vowels = 'aeiouyàâéèêëîïôùûüÿ'
#     english_vowel_count = sum(1 for char in message if char.lower() in vowels)

#     # Supposons que le message est en anglais si plus de 70% des caractères sont des voyelles
#     if english_vowel_count / len(message) > 0.7:
#         return 'en'  # Probablement en anglais

#     # Vérifier si le message contient plus de lettres avec des accents français
#     french_accents = 'àâéèêëîïôùûüÿ'
#     french_accent_count = sum(1 for char in message if char.lower() in french_accents)

#     # Supposons que le message est en français si plus de 70% des caractères ont des accents français
#     if french_accent_count / len(message) > 0.7:
#         return 'fr'  # Probablement en français

#     # Si aucune caractéristique spécifique n'est détectée, on suppose que c'est en anglais
#     return 'en'

# @app.route('/chatbot', methods=["GET", "POST"])
# def chatbotResponse():
#     if request.method == 'POST':
#         the_question = request.form['question']

#         # Detect the language of the user's input
#         language = detect_language(the_question)

#         # Use language-specific chatbot model and processing based on language
#         if language == 'fr':
#             response = procfr.obtenir_reponse(the_question)  # Chatbot français
#         else:
#             response = proc.chatbot_response(the_question)  # Chatbot anglais

#     return jsonify({"response": response})


cors = CORS(app, resources={"/chatbot": {"origins": "http://*"}})


@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():

    if request.method == 'POST':

        # the_question = request.form['question']

        # response = procfr.reponse_chatbot(the_question)

        input = request.get_json()
        question = input["question"]
        response = procfr.reponse_chatbot(question)
    return jsonify({"response": response})

# @app.route('/chatbot', methods=["GET", "POST"])
# def chatbotResponse():

#     if request.method == 'POST':
#         the_question = request.form['messageInput']

#         #response = procfr.reponse_chatbot(the_question)
#         response = procfr.reponse_chatbot(procfr.prevoir_classe("messageInput",procfr.model ), procfr.ints)
#     return jsonify({"chatput":response  })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
