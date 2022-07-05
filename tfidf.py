import pandas as pd
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')

algo = pd.read_json('sample.json')

algo['WordTokenize'] = algo['Description'] + algo['Title']

# Preprocessing Data
cnt = 0
for i in algo.WordTokenize:
    algo['WordTokenize'][cnt] = i.lower()
    cnt = cnt+1

algo.WordTokenize = algo.WordTokenize.replace(
    to_replace='lines:(.*\n)', value='', regex=True)
algo.WordTokenize = algo.WordTokenize.replace(
    to_replace='[.!"#$%&\'()*+,/:;<=>?@[\\]^_`{|}~]', value=' ', regex=True)    # Removing punctuation
algo.WordTokenize = algo.WordTokenize.replace(
    to_replace='-', value=' ', regex=True)
algo.WordTokenize = algo.WordTokenize.replace(
    to_replace='\s+', value=' ', regex=True)        # Removing new line
algo.WordTokenize = algo.WordTokenize.replace(
    to_replace='  ', value='', regex=True)          # Removing double white space
# Removing white space from beginning and end
algo.WordTokenize = algo.WordTokenize.apply(lambda x: x.strip())

# print(algo['WordTokenize'])

# Word Tokenization
cnt = 0
for i in algo.WordTokenize:
    algo['WordTokenize'][cnt] = word_tokenize(i)
    cnt = cnt+1

# print(algo['WordTokenize'])


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


algo['Keywords'] = wordLemmatizer(algo['WordTokenize'])
# print(algo['Keywords'])

# TF-IDF ALGORITHM

# Creating Vocabulary for unique words
vocabulary = set()
for doc in algo.Keywords:
    vocabulary.update(doc.split(','))
vocabulary = list(vocabulary)

# Intializating the tfIdf model
tfidf = TfidfVectorizer(vocabulary=vocabulary)
# Fit the TfIdf model
tfidf.fit(algo.Keywords)
# Transform the TfIdf model
tfidf_tran = tfidf.transform(algo.Keywords)

# print(tfidf)
# print(tfidf_tran)

# print('\nidf values:')
# for ele1, ele2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
#     print(ele1, ':', ele2)

# print('\nWord indexes:')
# print(tfidf.vocabulary_)

# # display tf-idf values
# print('\ntf-idf value:')
# print(tfidf_tran)

# # in matrix form
# print('\ntf-idf values in matrix form:')
# print(tfidf_tran.toarray())

with open('tfid.pkl', 'wb') as handle:
    pickle.dump(tfidf, handle)

with open('tfid_tran.pkl', 'wb') as handle:
    pickle.dump(tfidf_tran, handle)

with open("vocabulary.txt", "w") as file:
    file.write(str(vocabulary))
