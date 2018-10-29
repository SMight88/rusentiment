from flask import Flask, request, jsonify

from sentiment.sentiment import get_sentiment

app = Flask(__name__)


@app.route('/ready')
def ready():
    return 'OK'


@app.route('/', methods=['POST'])
def rusentiment():
    request_data = request.get_json()
    text = request_data['text']
    sentiment = get_sentiment(text)
    return jsonify({'class': sentiment})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
