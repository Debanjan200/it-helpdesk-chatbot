from flask import Flask,render_template,request
import nltk
import numpy as np
import re
from nltk.stem import LancasterStemmer
import pickle
import random
import json


nltk.download('punkt')
words,labels=pickle.load(open("data.pkl","rb"))
model=pickle.load(open("tree_Classifier_model.pkl","rb"))
stemmer=LancasterStemmer()
app=Flask(__name__)

with open("intents.json",encoding="utf-8") as f:
    data=json.load(f)

def bag_of_words(sen,words):
    bag=[0 for _ in range(len(words))]
    sen=re.sub("[^a-zA-Z]"," ",sen)
    s_words=nltk.word_tokenize(sen)
    s_words=[stemmer.stem(word.lower()) for word in s_words]

    for w in s_words:
        for i,w1 in enumerate(words):
            if w==w1:
                bag[i]=1

    return np.array(bag)

def prediction(s,words):
    bag=bag_of_words(s,words)
    tag=np.argmax(model.predict([bag]))
    tag=labels[tag]

    for tg in data["intents"]:
        if tg["tag"]==tag:
            response=tg["responses"]

    return random.choice(response)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/",methods=["POST"])
def predict():
    if request.method=="POST":
        text=(request.form["chatbot"])
    
    words,labels=pickle.load(open("data.pkl","rb"))
    pred=prediction(text,words)

    return render_template("index.html",prediction=pred)

if __name__=="__main__":
    app.run(debug=True)
