from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
import pickle
from helper.prediction import get_prediction

# Load the model and the scaler
with open('./models/LSTM/LSTM.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('./models/LSTM/tokenizer.pkl', 'rb') as model_file:
    tokenizer = pickle.load(model_file)

with open('./models/LSTM/vocab.pkl', 'rb') as model_file:
    vocab = pickle.load(model_file)

# Create the Flask app
app = Flask(__name__, static_folder='./static', static_url_path='')

# Enable CORS
CORS(app)

# Define the routes
@app.route('/')
def index_page():
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def serve_custom_path(path):
    return send_from_directory('./', path)

# This route will be used to predict the price of a car
@app.route('/predict', methods=['GET'])
def predict_price():
    input_search_text = request.args.get('search')

    prediction = get_prediction(model, tokenizer, vocab, prompt=input_search_text)

    return jsonify(prediction)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)