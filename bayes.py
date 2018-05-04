import json
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from sklearn.svm import SVR
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

X = evaluate(train_data)
Y = np.array(map(lambda doc: float(doc['sentiment']), train_data))

clf = SVR()
clf.fit(X, Y)
predicted_y = clf.predict(X)
diff = predicted_y - Y
train_mse = np.sum(diff*diff)/diff.shape[0]

test_x = evaluate(test_data)
test_y = np.array(map(lambda doc: float(doc['sentiment']), test_data))
test_result = clf.predict(test_x)
test_diff = test_result - test_y
test_mse = np.sum(test_diff*test_diff)/test_diff.shape[0]

print('train_mse', train_mse)
print('test_mse', test_mse)
