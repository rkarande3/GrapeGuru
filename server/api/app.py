from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
import os

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.info("HIT BACKEND")

@app.route('/api/process_data', methods=['POST'])
def process_data():
    try:
        data = request.json
        result = data.get('data') 
        if result is None:
            raise ValueError('Invalid JSON: Missing "data" field')
        logging.info(result)

        loaded_model = load_model('wine_recommender.h5')

        cleaned_df = pd.read_csv('datasets/wine_updated_ds.csv')
        df = cleaned_df.dropna(subset=['description', 'variety'])

        new_row = pd.DataFrame({'description': [result], 'variety': ['']})
        logging.info(len(df))
        df = pd.concat([df, new_row], ignore_index=True)
        logging.info("Dataset length after new row:")
        logging.info(len(df))

        # tokenize the input description
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df['description'])  # Pass a list of texts
        total_words = len(tokenizer.word_index) + 1

        # Convert text description to sequences
        sequences = tokenizer.texts_to_sequences(df['description'])
        padded_sequences = pad_sequences(sequences)
        logging.info(len(padded_sequences))
        logging.info(padded_sequences[-1])

        label_encoder = LabelEncoder()
        df['variety'] = label_encoder.fit_transform(df['variety'])

        # Make the prediction
        predictions = loaded_model.predict(padded_sequences[-1].reshape((1, -1)))
        #logging.info(len(predictions))

        # Convert the predicted probabilities to class labels
        predicted_labels = np.argmax(predictions, axis=1)

        # Decode the predicted labels back to original class names
        predicted_variety = label_encoder.inverse_transform(predicted_labels)

        
        logging.info("Processed data")
        logging.info(predicted_variety)
        variety = str(predicted_variety[0])
        logging.info(variety)

        return jsonify({"message": variety})

    except Exception as e:
        logging.info(f"Error processing data: {e}")
        return jsonify({'error': 'An error occurred during processing'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
