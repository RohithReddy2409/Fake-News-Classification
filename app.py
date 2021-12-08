import tkinter as tk  
from functools import partial  
   
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from scipy.sparse import hstack
from scipy import sparse

#Reference- https://www.javatpoint.com/python-tkinter-text

text_vectorizer=pd.read_pickle("text_vectorizer.pickle")
author_vectorizer=pd.read_pickle("author_vectorizer.pickle")
title_vectorizer=pd.read_pickle("title_vectorizer.pickle")
naive_bayes=pd.read_pickle("gridsearch_naive_bayes.pickle")

def preprocess_sentence(sentence):
    #Lowecase conversion and removing spaces at beginning & end
    preprocessed_sentence= sentence.lower().strip()

    #Remove special characters-Keep only alpha numeric characters and spaces between words
    preprocessed_sentence=re.sub('[^A-Za-z0-9 ]+', '', preprocessed_sentence)


    stop_words = set(stopwords.words('english'))  
    word_tokens = preprocessed_sentence.split()
    
    #Removing stopwords
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    filtered_sentence=" ".join(filtered_sentence)
    
    return filtered_sentence

   
def call_result(label_result, input_title, input_author, input_text):  

    title= input_title.get()  
    author= input_author.get()
    text= input_text.get()
    print("******************Before preprocessing******************")
    print("Title:",title)
    print("Author:",author)
    print("Text:",text)
    
    preprocessed_title=preprocess_sentence(title)
    preprocessed_author=preprocess_sentence(author)
    preprocessed_text=preprocess_sentence(text)
    
    
    print("******************After preprocessing******************")
    print("Title:",preprocessed_title)
    print("Author:",preprocessed_author)
    print("Text:",preprocessed_text)
    
    #ext="hello"--> ["hello"]
    
    preprocessed_title=[preprocessed_title]
    preprocessed_author=[preprocessed_author]
    preprocessed_text=[preprocessed_text]
    
    #r1,r2,r3]
    X_test_title=title_vectorizer.transform(preprocessed_title)
    
    #(1,10)
    
    #print(X_test_title,X_test_title.shape)
    X_test_author=author_vectorizer.transform(preprocessed_author)
    
    #(1,5)
    #print(X_test_author,X_test_author.shape)
    X_test_text=text_vectorizer.transform(preprocessed_text)
    #print(X_test_text,X_test_text.shape)
    #(1,500)
    
    
    X_test_sparse=hstack((sparse.csr_matrix((X_test_text)),X_test_title,X_test_author))
    
    #(1,515)
    
    #(4,515)
    
    prediction=naive_bayes.predict(X_test_sparse.todense()[0])
    #utput-->[0] / [1]
    
    print("Predicted label:", prediction[0])
    
    if prediction[0]==0:
        print("Reliable news")
        predicted_label="Reliable news"
    elif prediction[0]==1:
        print("Unreliable news")
        predicted_label="Unreliable news"

    label_result.config(text="Prediction: %s" % (predicted_label))  
    return  
   
root = tk.Tk()  
root.geometry('500x500')  
  
root.title('Fake News Classification')  
   
input_title = tk.StringVar()  
input_author= tk.StringVar()  
input_text= tk.StringVar()  

  
labelNum1 = tk.Label(root, text="Enter Title").grid(row=1, column=0)  
  
labelNum2 = tk.Label(root, text="Enter Author").grid(row=2, column=0)  

labelNum3 = tk.Label(root, text="Enter Text").grid(row=3, column=0)  
  

labelResult = tk.Label(root)  
  
labelResult.grid(row=7, column=1)  
  
entryNum1 = tk.Entry(root, textvariable=input_title, width=100).grid(row=1, column=2)  
  
entryNum2 = tk.Entry(root, textvariable=input_author,width=100).grid(row=2, column=2)  

entryNum3 = tk.Entry(root, textvariable=input_text,width=100).grid(row=3, column=2)  

  
call_result = partial(call_result, labelResult, input_title, input_author, input_text)  
  
buttonCal = tk.Button(root, text="Classify", command=call_result).grid(row=4, column=0)  
  
root.mainloop()  