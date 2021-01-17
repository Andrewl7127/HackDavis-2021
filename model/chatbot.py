#import pyttsx3


#engine = pyttsx3.init()
#engine.say("I will speak this text")
#engine.runAndWait()

import nltk
from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer()

import json
import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle
import pandas as pd

with open("intents.json") as file:
    data = json.load(file)

try:
   with open("data1.pickle", "rb") as f:
       words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    for intents in data['intents']:
        for pattern in intents['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intents['tag'])
            
            if intents['tag'] not in labels:
                labels.append(intents['tag'])
    
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)
    
    training = []
    output = []
    
    out_empty = [0 for _ in range(len(labels))]
    
    for x,doc in enumerate(docs_x):
        bags = []
        wrds = [lemmatizer.lemmatize(w) for w in doc]
        
        for w in words:
            if w in wrds:
                bags.append(1)
            else:
                bags.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bags)
        output.append(output_row)
        
    training = np.array(training)
    output = np.array(output)
    with open("data.pickle", "wb") as f:
       pickle.dump((words, labels, training, output), f)


tf.compat.v1.reset_default_graph() 

nn = tflearn.input_data(shape = [None, len(training[0])])
nn = tflearn.embedding(nn, input_dim=20000, output_dim=128, trainable=False, name="EmbeddingLayer")
nn = tflearn.fully_connected(nn, 8)
nn = tflearn.fully_connected(nn, 16)
nn = tflearn.fully_connected(nn, len(output[0]), activation = "softmax")
nn = tflearn.regression(nn)

model = tflearn.DNN(nn)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch = 1000, batch_size = 4, show_metric = True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]
    
    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1
    nparray = np.array(bag)
    return(nparray.reshape(-1,len(training[0])))

def chat():
    print("Hi there, ask any questions you may have about COVID-19 (type 'quit' to stop")
    data_15 = pd.DataFrame() # change this to  the actual df
    data_15['cases'] = [3,2,6,4]
    data_15['deaths'] = [3,9,7,7]
    data_14 = pd.DataFrame() # change this to the actual df
    data_14['cases'] = [1,2,3,4]
    data_14['deaths'] = [3,5,6,7]
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        results = model.predict(list(bag_of_words(inp, words)))[0]
        results_index = np.argmax(results)
        tag = labels[results_index]
        if results[results_index] > 0.5:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                
            if tag == 'cases':
                print("There have been {} new cases in the United States today, and {} new deaths".format(sum((data_15-data_14)['cases']), sum((data_15-data_14)['cases']))) 
            else:
                print(random.choice(responses))
        else:
            print("I didn't quite get that, please try again or ask a different question")

chat()

    
