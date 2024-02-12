from flask import Flask, request, jsonify,render_template
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('french')
import procfr


app = Flask(__name__)

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('chat.html', **locals())





@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():

    if request.method == 'POST':
        the_question = request.form['question']

        response = procfr.obtenir_reponse(the_question,)

    return jsonify({"response": response })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    