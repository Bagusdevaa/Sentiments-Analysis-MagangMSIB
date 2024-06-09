import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# Memakai pipeline untuk lebih mudal labelling
from transformers import pipeline



def preprocessing(data) -> list: 
    # set-up stop words bahasa indonesia dengan NLTK
    stop_words = set(stopwords.words("indonesian"))

    # define stemmer
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()

    # Mengubah ke huruf kecil semua, menghapus beberapa karakter tidak penting, dan menghapus punctuation
    df_lower = []
    for text in data:
        cleaned = text.lower().replace("'",'').replace('\t','').replace('"','').replace('\xa0','').strip()
        df_lower.append(cleaned.translate(str.maketrans('\n',' ',string.punctuation)))

    # Melakukan stopwords
    df_sw = []
    for word in df_lower:
        tokenize = word.split() # tokenisasi tiap kalimat
        cleaned_word = [i for i in tokenize if i not in stop_words]
        cleaned_sentence = ' '.join(cleaned_word)
        df_sw.append(cleaned_sentence)

    # Melakukan Stemming
    df_st = []
    for word in df_sw:
        tokens = word.split()
        tokens = [stemmer.stem(i) for i in tokens]

        # Removing Non-Alphanumeric Characters (jika ada karakter non-alfanumerik)
        tokens = [j for j in tokens if j.isalnum()]

        final = ' '.join(tokens)
        df_st.append(final)
    
    df = pd.DataFrame(df_st)
    return df


def PredictDataFrame(data) -> pd.DataFrame:
    pipe = pipeline("text-classification", model="ayameRushia/roberta-base-indonesian-1.5G-sentiment-analysis-smsa")
    tes = {}
    idx = 0
    for text in data:
        try:
            tes[idx] = pipe(text)[0]
        except:
            print("can't process line  ", idx)
        idx += 1
    df = pd.DataFrame(tes).T
    df = pd.concat([data, df], axis=1)

    return df
    

