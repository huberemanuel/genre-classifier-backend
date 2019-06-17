from flask import Flask, request
from flask_cors import CORS
import json
import pickle
from preprocess import pre_process, bow_transform
import numpy as np

app = Flask(__name__)
CORS(app)
c_sinonimo = 1 # sem sinonimo
metodo = 1 #com metodo postag
filename = './model.sav'
tok_name = './tokenizer.sav'
model = pickle.load(open(filename, 'rb'))
tokenizer = pickle.load(open(tok_name, 'rb'))
classes = [
    'Arts & Entertainment & Culture',
    'Economy & Business & Financial',
    'Politics',
    'Science & Technology',
    'Sport'
]

@app.route('/')
def classify():
    global tokenizer, model, classes
    question = request.args.get('question')
    question = pre_process( np.array([question]), c_sinonimo, metodo )
    question = bow_transform( question, tokenizer )
    #ret = model.predict(np.array([question]))
    ret = model.predict(question)
    # print(ret)
    return json.dumps(classes[int(ret[0])])
