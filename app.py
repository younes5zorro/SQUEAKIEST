import re
import string
import random
import datetime
import codecs
import json
import itertools

from flask import Flask, render_template, request, jsonify , session, url_for, redirect

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle


from keras.models import load_model
from keras.preprocessing import  sequence
app = Flask(__name__)


with open('mapping.manual.json',encoding='utf-8') as f:
    en_to_ar = json.loads(f.read())

NWORDS = {}
result = pd.read_csv("result.txt",sep=' ') 
result.columns = ['word', 'val']

for index, row in result.iterrows():
    NWORDS[row['word']] = int(row['val'])

def sort_by_frequency(words):
    sor = {}
    for w in words:
        sor[w] = NWORDS.get(w,1)
    return sorted(words, key=sor.__getitem__, reverse=True)

def transliterate_word(english):
  ret = set()

  def recur(letters, word, start=False):
    if len(letters) == 0:
      ret.add(word)
      return
    if start:
      table = en_to_ar['start']
    else:
      table = en_to_ar['other']
    max_key_len = len(max(list(table), key=len))
    for i in range(1, max_key_len + 1):
      l = letters[:i]
      if l in table:
        for ar in table[l]:
          recur(letters[i:], word + ar)

  recur(english, '', True)
  return ret

def transliterate(sentence, verbose=False):
    sentence = clear(sentence)
    words = sentence.split()
    ret = []
    for word in words:
        try: 
            candidates = list(transliterate_word(word))
            if verbose:
                for word in candidates:
                    print (word)
            ret.append(sort_by_frequency(candidates)[0])
        except IndexError:
            ret.append(word)

    return ' '.join(ret)

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

def clear(sentence):
    sentence = emoji_pattern.sub(r'',sentence)
    sentence = re.sub(r'^RT', '', sentence)
    sentence = re.sub(r'\&\w*;', '', sentence)
    sentence = re.sub(r'\@\w*', '', sentence)
    sentence = re.sub(r'\$\w*', '', sentence)
    sentence = re.sub(r'https?:\/\/.*\/\w*', '', sentence)
    sentence = re.sub(r'#\w*', '', sentence)
    sentence = re.sub(r'[.]+', '', sentence)
    sentence = re.sub(r'[\t,\.!?`()\-$*&؟\"\\\'،%~§+]', '', sentence)        
    sentence = re.sub(r'[»]', ' ', sentence)
    sentence = sentence.lower()
    sentence = sentence.strip(' ')
    sentence = re.sub('  ', ' ', sentence)
    sentence = ''.join(ch for ch, _ in itertools.groupby(sentence))
    return sentence


# label encode the target variable 
encoder = preprocessing.LabelEncoder()
encoder.fit(['egyptian', 'general', 'gulf', 'iraqi', 'levantine', 'maghrebi', 'tunisien'])

def page_acceuil():
    return render_template('home1.html')

def Modeling(txt,vec, model):
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("models/"+vec+".pkl", "rb")))
    loaded_model = pickle.load(open("models/"+model+".pkl", 'rb'))

    value = encoder.inverse_transform(loaded_model.predict(loaded_vec.transform([txt])))

    return value[0]

def Boosting(txt,vec, model):
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("models/"+vec+".pkl", "rb")))
    loaded_model = pickle.load(open("models/"+model+".pkl", 'rb'))

    value = encoder.inverse_transform(loaded_model.predict((loaded_vec.transform([txt])).tocsc()))

    return value[0]

def Deeplearning(txt,vec,model):
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("models/"+vec+".pkl", "rb")))
    loaded_model = load_model("models/"+model+".h5")
    predictions = loaded_model.predict(loaded_vec.transform([txt]))
    predictions = predictions.argmax(axis=-1)
    value = encoder.inverse_transform(predictions)

    return value[0]

def LSTM(txt,vec,model):

    tokenizer = pickle.load(open("models/"+vec+".pkl", "rb"))

    valid_seq_x = sequence.pad_sequences(tokenizer.texts_to_sequences([txt]), maxlen=70)

    loaded_model = load_model("models/"+model+".h5")
    predictions = loaded_model.predict(valid_seq_x)
    predictions = predictions.argmax(axis=-1)
    value = encoder.inverse_transform(predictions)

    return value[0]


def test(id,txt):
    value =""
    cleared = clear(txt)
    txt = transliterate(cleared)

    if(id == "1"):
        value = Modeling(txt,"count_vect","NB1_CountVectors")
    elif(id == "2"):
        value = Modeling(txt,"Tfidf_word_vect","NB2_WordLevelTFIDF")
    elif(id == "3"):
       value = Modeling(txt,"tfidf_vect_ngram","NB3_NGramVectors")
    elif(id == "4"):
       value = Modeling(txt,"tfidf_vect_ngram_chars","NB4_CharLevelVectors")

    elif(id == "5"):
        value = Modeling(txt,"count_vect","LR1_CountVectors")
    elif(id == "6"):
        value = Modeling(txt,"Tfidf_word_vect","LR2_WordLevelTFIDF")
    elif(id == "7"):
        value = Modeling(txt,"tfidf_vect_ngram","LR3_NGramVectors")
    elif(id == "8"):
        value = Modeling(txt,"tfidf_vect_ngram_chars","LR4_CharLevelVectors")
    
    elif(id == "9"):
        value = Boosting(txt,"count_vect","Xgb1_CountVectors")
    elif(id == "10"):
        value = Boosting(txt,"Tfidf_word_vect","Xgb2_WordLevelTFIDF")    
    elif(id == "11"):
        value = Boosting(txt,"tfidf_vect_ngram_chars","Xgb3_CharLevelVectors")   

    elif(id == "12"):
        value = Deeplearning(txt,"tfidf_vect_ngram","NN_NgramLevelTFIDFVectors")
    elif(id == "13"):
        value = LSTM(txt,"fit_on_texts","CNN_WordEmbeddings")
    elif(id == "14"):
        value = LSTM(txt,"fit_on_texts","RNN_LSTM_WordEmbeddings")
    elif(id == "15"):
        value = LSTM(txt,"fit_on_texts","RNN_GRU_WordEmbeddings")
    elif(id == "16"):
        value = LSTM(txt,"fit_on_texts","CNN_WordEmbeddings")

    elif(id == "17"):
        value = Modeling(txt,"count_vect","SVM1_CountVectors")
    elif(id == "18"):
        value = Modeling(txt,"Tfidf_word_vect","SVM2_WordLevelTFIDF")
    elif(id == "19"):
       value = Modeling(txt,"tfidf_vect_ngram","SVM3_NGramVectors")

    elif(id == "20"):
       value = Modeling(txt,"count_vect","RF1_CountVectors")
    elif(id == "21"):
       value = Modeling(txt,"Tfidf_word_vect","RF2_WordLevelTFIDF")
    else:
        value = "Parametres Error ...."
    doc = {
        "txt":txt,
        "cleared":cleared,
        "id":id,
        "value":value
        }
    
    return jsonify(doc)


def clean(txt):
    value = clear(txt)
    trns = transliterate(value)
    doc = {
        "txt":txt,
        "value":value,
        "tanslat":trns
        }
    
    return jsonify(doc)


# routes app
app.add_url_rule('/', 'index', page_acceuil, methods=['GET'])


app.add_url_rule('/api/get/<id>/<txt>', 'test', test, methods=['GET'])
app.add_url_rule('/api/clean/<txt>', 'clean', clean, methods=['GET'])

if __name__ == "__main__":
    app.run(debug=True)

