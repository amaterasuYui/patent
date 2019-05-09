%load_ext autoreload
%autoreload 2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from patent.doc import PatentDoc
from patent.patentmodel import PatentModel
from matplotlib import pyplot
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

useful_abs_cut = patent.cut_words(useful_abs, 1)
useless_abs_cut = patent.cut_words(useless_abs, 1)

all_docs = useful_abs_cut + useless_abs_cut
label = np.concatenate([np.repeat(1, len(useful)), 
                        np.repeat(0, len(useless))])

doc_bow, doc_tfidf, doc_dict, _ = patent.bow_tfidf_matrix(all_docs)

#------------------predict by bow ---------------
train_x, test_x, train_y, test_y = train_test_split(doc_bow, label, test_size = 0.3, random_state = 88)
bow_mdl = PatentModel(train_x, train_y, test_x, test_y)
bow_lr_mdl, bow_lr_test_report = bow_mdl.logistic_reg()
bow_nb_mdl, bow_nb_test_report = bow_mdl.naive_bayes()
bow_svm_mdl, bow_svm_test_report = bow_mdl.svm()
bow_nn_mdl, bow_nn_test_report = bow_mdl.neural_network()

#----------------predict by tfidf------------------
train_x, test_x, train_y, test_y = train_test_split(doc_tfidf, label, test_size = 0.3, random_state = 88)
tfidf_mdl = PatentModel(train_x, train_y, test_x, test_y)
tfidf_lr_mdl, tfidf_lr_test_report = tfidf_mdl.logistic_reg()
tfidf_nb_mdl, tfidf_nb_test_report = tfidf_mdl.naive_bayes()
tfidf_svm_mdl, tfidf_svm_test_report = tfidf_mdl.svm()
tfidf_nn_mdl, tfidf_nn_test_report = tfidf_mdl.neural_network()

# plot roc curve
bow_mdl.roc_curve_plot(bow_nb_mdl, "BOW NB ROC curve")
pyplot.show()

bow_mdl.roc_curve_plot(bow_nn_mdl, "BOW NN ROC curve", True)
pyplot.show()

