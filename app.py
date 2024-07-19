import numpy as np
import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras import models

# Load the LSTM model
model = models.load_model('next_word_lstm.h5')

## load the tokenizer
with open('tokenizer.pickle' ,'rb') as handle:
    tokenizer = pickle.load(handle)


def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    #if len(token_list) >= max_sequence_len:
    token_list = token_list[-(max_sequence_len):]
    token_list = pad_sequences([token_list] ,maxlen=max_sequence_len , padding='pre')
    predicted =model.predict(token_list,verbose=0)
    predict_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predict_word_index:
            return word
        
# streamlit app

st.title("Next Word Prediction with LSTM and Early Stopping")
input_text = st.text_input("Enter the Sequence of Words" ,"To be or not to be")
if st.button("Predict next Word"):
    max_sequence_len = len(list(input_text))+1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(next_word)


