import json
import pandas as pd
import numpy as np
import re
import nltk
import pickle
import sys
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')

tfidf_tran = pickle.load(open('tfid_tran.pkl', 'rb'))
tfidf = pickle.load(open('tfid.pkl', 'rb'))

algo = pd.read_json('sample.json')
with open("vocabulary.txt", "r") as file:
    vocabulary = eval(file.readline())

# Function to map find out the correct POS tag for word, map it to the right input character that the WordnetLemmatizer accepts


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wn.ADJ,
                "N": wn.NOUN,
                "V": wn.VERB,
                "R": wn.ADV}

    return tag_dict.get(tag, wn.NOUN)


def wordLemmatizer(data):           # Function for lemmatizing the word
    result = pd.DataFrame()
    for index, entry in enumerate(data):
        Final_words = []    # List to store the final lemmatized list of words of a sentence index
        lemmatizer = WordNetLemmatizer()
        for word, tag in pos_tag(entry):
            # Checking length of word and is present in stopword and is alphabet
            if len(word) > 1 and word not in stopwords.words('english') and word.isalpha():
                word_Final = lemmatizer.lemmatize(word, get_wordnet_pos(word))
                Final_words.append(word_Final)
                result.loc[index, 'Keyword_final'] = str(Final_words)
                result = result.replace(
                    to_replace="\[.", value='', regex=True)
                result = result.replace(
                    to_replace="'", value='', regex=True)
                result = result.replace(
                    to_replace=" ", value='', regex=True)
                result = result.replace(
                    to_replace='\]', value='', regex=True)
    return result


def gen_vector_T(tokens):
    Q = np.zeros((len(vocabulary)))
    x = tfidf.transform(tokens)
    for token in tokens[0].split(','):
        try:
            ind = vocabulary.index(token)
            Q[ind] = x[0, tfidf.vocabulary_[token]]
        except:
            pass
    return Q


def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim


def cosine_similarity_T(k, query):
    preprocessed_query = preprocessed_query = re.sub("\W+", " ", query).strip()
    tokens = word_tokenize(str(preprocessed_query))
    q_df = pd.DataFrame(columns=['q_clean'])
    q_df.loc[0, 'q_clean'] = tokens
    q_df['q_clean'] = wordLemmatizer(q_df.q_clean)
    d_cosines = []

    query_vector = gen_vector_T(q_df['q_clean'])
    for d in tfidf_tran.A:
        d_cosines.append(cosine_sim(query_vector, d))

    out = np.array(d_cosines).argsort()[-k:][::-1]
    d_cosines.sort()
    a = pd.DataFrame()
    for i, index in enumerate(out):
        a.loc[i, 'index'] = str(index)
        a.loc[i, 'URL'] = algo['URL'][index]
        a.loc[i, 'Title'] = algo['Title'][index]
        a.loc[i, 'Description'] = algo['Description'][index]
    for j, simScore in enumerate(d_cosines[-k:][::-1]):
        a.loc[j, 'Score'] = simScore
    return a


x = cosine_similarity_T(10, sys.argv[1])
x = x.to_json()
print(x)
sys.stdout.flush()
