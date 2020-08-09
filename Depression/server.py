import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import nltk
import json
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

app = Flask(__name__)

weights_file = 'weights.json' 
ERROR_THRESHOLD = 0.1



with open(weights_file) as data_file: 
    weights = json.load(data_file) 
    W1 = np.asarray(weights['weight1']) 
    W2 = np.asarray(weights['weight2'])
    b1 = np.asarray(weights['bias1']) 
    b2 = np.asarray(weights['bias2'])
    all_words = weights['words']
    classes = weights['classes']

def encode_sentence(all_words,sentence, bag):
    for s in sentence:        
        stemmed_word = stemmer.stem(s)
        for i,word in enumerate(all_words):
            if stemmed_word == word:
                bag[i] = 1
    return bag

# Method for calculating relu
def relu(z):
    A = np.array(z,copy=True)
    A[z<0]=0
    assert A.shape == z.shape
    return A

# Method for calculating softmax
def softmax(x):
    num = np.exp(x-np.amax(x,axis=0,keepdims=True))    
    return num/np.sum(num,axis=0,keepdims=True)

def clean_sentence(verification_data):
    line = verification_data
    # Remove whitespace from line and lower case iter
    line = line.strip().lower()
    # Removing word with @ sign as we dont need name tags of twitter
    line = " ".join(filter(lambda x:x[0]!='@', line.split()))
    # Remove punctuations and numbers from the line
    punct = line.maketrans("","",'.*%$^0123456789#!][\?&/)/(+-<>')
    result = line.translate(punct)
    # Tokenize the whole tweet sentence
    tokened_sentence = nltk.word_tokenize(result)
    # We take the tweet sentence from tokened sentence
    sentence = tokened_sentence[0:len(tokened_sentence)]
    return sentence    

def verify(sentence, show_details=False):
    bag=[0]*len(all_words)
    cleaned_sentence = clean_sentence(sentence)
    # This line returns the bag of words as 0 or 1 if words in sentence are found in all_words
    x = encode_sentence(all_words,cleaned_sentence,bag)
    x = np.array(x)
    x = x.reshape(x.shape[0],1)
    
#    print("Shape of X is ", x.shape)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our encoded sentence
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = relu(np.dot(W1,l0)+b1)
    # output layer
    l2 = softmax(np.dot(W2,l1)+b2)
    
    return l2

def classify(sentence, show_details=False):
    results = verify(sentence, show_details)
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes[r[0]],r[1]] for r in results]
    # print ("%s \n classification: %s \n" % (sentence, return_results[0][0]))
    # return return_results
    return "{ sentence : %s , classification: %s }" % (sentence, return_results[0][0])


@app.route('/')
def home():
    return jsonify({})

@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()
    return jsonify(classify(data['sentence']))


if __name__ == "__main__":
    app.run(debug=True)