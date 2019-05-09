%load_ext autoreload
%autoreload 2
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from gensim.models import Word2Vec
from patent.doc import PatentDoc
import warnings
warnings.filterwarnings("ignore")

useful_path = "data/useful.CSV"
useless_path = "data/useless.CSV"
stop_words_path = "data/chinese_stop_words.txt"

patent = PatentDoc(useful_path, useless_path, stop_words_path)
useful = patent.useful_docs
useless = patent.useless_docs

useful_abs = (useful.摘要 + useful.第一权利要求).tolist()

useless_abs= (useless.摘要 + useless.第一权利要求).tolist()

useful_abs_cut = patent.cut_words(useful_abs)
useless_abs_cut = patent.cut_words(useless_abs)

# create word2vec model
all_docs = useful_abs_cut + useless_abs_cut
word2vec_mdl = Word2Vec(all_docs,
                        size = 100,
                        iter = 10,
                        min_count = 20)

# get doc vectors
doc_vec = patent.text_score_vector(all_docs, word2vec_mdl)

# train classification model
label = np.concatenate([np.repeat(1, len(useful)), 
                        np.repeat(0, len(useless))])
labelEncoder = LabelEncoder()
label = labelEncoder.fit_transform(label)
train_x, test_x, train_y, test_y = train_test_split(doc_vec, label, test_size = 0.3)
lr_mdl = LogisticRegression()
lr_mdl.fit(train_x, train_y)
lr_mdl.score(test_x, test_y)

print(classification_report(train_y, lr_mdl.predict(train_x)))




#------------------predict by bow
from sklearn.feature_extraction import DictVectorizer
from collections import Counter
v = DictVectorizer()

all_docs_words = v.fit_transform(Counter(doc) for doc in all_docs)
train_x, test_x, train_y, test_y = train_test_split(all_docs_words, label, test_size = 0.3)
lr_mdl = LogisticRegression()
lr_mdl.fit(train_x, train_y)
lr_mdl.score(test_x, test_y)

print(classification_report(test_y, lr_mdl.predict(test_x)))


