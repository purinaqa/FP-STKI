import os
import re
import unicodedata
import mysql.connector
import numpy as np
import errno
import sys
import tempfile
import random
import json
from keras.models import load_model

static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')
db = mysql.connector.connect(user="root", password='', database="glove")
cursor = db.cursor(buffered=True)
model = load_model("data-target-model-wtl.h5")

def getWordEmbedding(word, cursor):
#     word = word.replace("'", "''")
    sql = """select vec from term where term like %s"""
    cursor.execute(sql, (str(word),))
    data = cursor.fetchall()
    if len(data) > 0:
        decoded_vec = json.JSONDecoder().decode(data[0][0])
        vec = np.asarray(decoded_vec, dtype=np.float32)
        return True, vec
    else:
        return False, data

def myTokenizer(content, lower=True):
    raw = content.split(' ')
    remover = re.compile("[^a-zA-Z-]")

    token = []

    for i in raw:
        term = remover.sub('', i)
        if lower == True:
            term = term.lower()
        token.append(term)
    tokenized = list (filter(None, token))

    return tokenized

def toSentenceEmbd(string):
    string = string.replace('\n', '')
    string = np.array(myTokenizer(string))

    begin = True
    for word in string:
        stat, vec = getWordEmbedding(word, cursor)
        if not stat:
            continue
        if begin:
            begin = False
            feature = vec
        else:
            feature += vec
            # feature = np.concatenate([feature, vec])

    feature = feature/np.linalg.norm(feature)
    feature = np.array(feature)

    return feature

def getPrediction(doc):
    vec = toSentenceEmbd(doc)
    vec = vec.reshape((1, 300, 1))
    prediction = model.predict([vec])[0]
    print(prediction)
    argmax = np.argmax(prediction)
    if prediction[argmax] < 0.7:
        print("DOUBT")
        ids = []
        for i in range(3):
            prediction[argmax] = -1
            ids.append(argmax+1)
            argmax = np.argmax(prediction)
        ids.append(23)
        return ids
    else:
        return [argmax+1]

def getAnswer(dictionary):
    dictionary = str(dictionary)
    with open('data_islamicQA.json', 'r') as data_file:
        data = json.load(data_file, strict=False)
    return random.choice(data[dictionary])

def make_static_tmp_dir():
    try:
        os.makedirs(static_tmp_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(static_tmp_path):
            pass
        else:
            raise

if __name__ == '__main__':
    while True:
        user_input = input('input: ')
        id = getPrediction(user_input)
        answer = getAnswer(id[0])
        print('answer: {}'.format(answer))