import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


import re
import joblib
import json

# load encoder
encoder = joblib.load('encoder/encoder.joblib')

# load tokenizer
with open('tokenizer/tf_tokenizer.json', 'r') as f:
    tokenizer_json = json.load(f)

tokenizer = tokenizer_from_json(tokenizer_json)


# load model
model = tf.keras.models.load_model('models/starter_swahili_news_classification_model.h5')

# normalize text
max_words = 1000  # fetched from tokenizer
max_len = 200      # fetched from tokenizer

def normalize_text(text):
    # Remove punctuation, numbers, and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def pre_process(tokenizer, max_len, input_text):
    input_text = normalize_text(input_text)
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_data = pad_sequences(input_sequence, maxlen=max_len)
    return input_data


def classify_news(input_text):
    input_data = pre_process(tokenizer, max_len, input_text)
    pred = model.predict(input_data)
    # for each input sample, the model returns a vector of probabilities
    # return all classes with their corresponding probabilities
    result_dict = {}

    for i, category in enumerate(encoder.categories_[0]):
        result_dict[category] = str(round(pred[0][i] * 100, 2))+'%'

    highest_prob = max(result_dict, key=result_dict.get)

    return (result_dict, highest_prob)
