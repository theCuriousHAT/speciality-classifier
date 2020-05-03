#import pickle
import json
import numpy as np
#from sklearn.externals import joblib
#from sklearn.linear_model import Ridge
from azureml.core.model import Model
import joblib
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer
from textblob import Word
from nltk.tokenize import word_tokenize
import re

from sklearn import linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def init():
    global model,vec_model
    # note here "best_model" is the name of the model registered under the workspace
    # this call should return the path to the model.pkl file on the local disk.
    model_path1 = Model.get_model_path(model_name='outputs/spe_model.pkl')
    model_path2 = Model.get_model_path(model_name ='outputs/tfid_vectors.pkl')
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path1)
    vec_model = joblib.load(model_path2)


# note you can pass in multiple rows for scoring
def run(raw_data):
    global text
    try:
        text = str(json.loads(raw_data)['data'])
        text = text.lower()     # Converting to lowercase
        text = re.sub(r'[?|!|\'|"|#|,|)|(|\|/$%\n\t.:;""‘’]',r'',text)
        tfidf_vect = vec_model.transform([text])
        #data = np.array(data)
        result = model.predict(tfidf_vect)
        encoder_name_mapping={0:'Cardiology',1:'Family medicine/general practice',2:'Internal medicine',3:'Neurology',
                        4:'Obstetrics / gynecology',5:'Orthopedics',6:'Others',7:'Radiology',8: 'Surgery',
                        9:'Urology'}
        category = encoder_name_mapping.get(result[0])

        # you can return any data type as long as it is JSON-serializable
        
        return category.tolist()
    except Exception as e:
        category = str(e)
        return category