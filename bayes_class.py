import json
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import numpy as np
import re

train_file = open('./training_set.json', 'r')
test_file = open('./test_set.json', 'r')

train_data = json.loads(train_file.read())
test_data = json.loads(test_file.read())


tokenizer = TweetTokenizer()
stemmer = LancasterStemmer()
stop = set(stopwords.words('english'))

def get_n_grams(s, n=1):
    tokens = [w for w in map(lambda x: x.lower(), tokenizer.tokenize(s)) if w not in stop]
    n_grams = []
    for i in range(len(tokens) - n + 1):
        n_grams.append(tokens[i:i+n])
    return n_grams

def get_stemmed_n_grams(s, n=1):
    tokens = [w for w in map(stemmer.stem, tokenizer.tokenize(s)) if w not in stop]
    n_grams = []
    for i in range(len(tokens) - n + 1):
        n_grams.append(tokens[i:i+n])
    return n_grams

# initialize count by features

score = {}
count = {}

feature_list = [
    {
        'name': 'unigram',
        'tokenizer': get_n_grams,
        'n': 1
    },
    {
        'name': 'bigram',
        'tokenizer': get_n_grams,
        'n': 2
    },
    {
        'name': 'stemmed_unigram',
        'tokenizer': get_stemmed_n_grams,
        'n': 1
    },
    {
        'name': 'stemmed_bigram',
        'tokenizer': get_stemmed_n_grams,
        'n': 2
    }
]

for feature in feature_list:
    score[feature['name']] = {}
    count[feature['name']] = {}

# counting
for doc in train_data:
    if type(doc['snippet']) == unicode:
        doc['snippet'] = [doc['snippet']]
    for text in doc['snippet']:
        proc = re.sub('\d+.*', '<NUM>', text)
        proc = re.sub(r'\+<NUM>.*', '<+NUM>', proc)
        proc = re.sub(r'\+<NUM>.*', '<+NUM>', proc)
        for feature in feature_list:
            n_grams = feature['tokenizer'](proc, n=feature['n'])
            for n_gram in n_grams:
                tup = tuple(n_gram)
                if tup not in score[feature['name']]:
                    score[feature['name']][tup] = 0
                    count[feature['name']][tup] = 0
                score[feature['name']][tup] += float(doc['sentiment'])
                count[feature['name']][tup] += 1

def evaluate(docs):
    X = []
    for doc in docs:
        x = []
        if type(doc['snippet']) == unicode:
            doc['snippet'] = [doc['snippet']]
        for feature in feature_list:
            for text in doc['snippet']:
                v = 0.0
                n_grams = feature['tokenizer'](text, n=feature['n'])
                for n_gram in n_grams:
                    tup = tuple(n_gram)
                    if tup in score[feature['name']] and count[feature['name']][tup] > 1: v += score[feature['name']][tup]
            x.append(v/max(1, len(n_grams)))
        X.append(x)
    return np.array(X)

def to_class(value):
    if abs(value) < 1e-5: return 0
    elif value > 0: return 1
    else: return 2
X = evaluate(train_data)
Y = np.array(map(lambda doc: to_class(float(doc['sentiment'])), train_data))

clf = SVC()
clf.fit(X, Y)
predicted_y = clf.predict(X)
train_macro_f1 = f1_score(Y, predicted_y, average='macro')
train_micro_f1 = f1_score(Y, predicted_y, average='micro')
print('training')
print('macro-f1 = ' + str(train_macro_f1))
print('micro-f1 = ' + str(train_micro_f1))

test_x = evaluate(test_data)
test_y = np.array(map(lambda doc: to_class(float(doc['sentiment'])), test_data))
test_result = clf.predict(test_x)

test_macro_f1 = f1_score(test_y, test_result, average='macro')
test_micro_f1 = f1_score(test_y, test_result, average='micro')
print('testing')
print('macro-f1 = ' + str(test_macro_f1))
print('micro-f1 = ' + str(test_micro_f1))
