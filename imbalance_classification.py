%load_ext autoreload
%autoreload 2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SVMSMOTE
from patent.doc import PatentDoc
from patent.patentmodel import PatentModel
from matplotlib import pyplot
import warnings
warnings.filterwarnings("ignore")

###### create useless csv
# useful_data = pd.read_csv("data/DC/useful.CSV")
# usefulless_data = pd.read_csv("data/DC/useless_useful.CSV")
# useless_data = usefulless_data[~usefulless_data["公开(公告)号"].isin(useful_data["公开(公告)号"])]
# useless_data.to_csv("data/DC/useless.CSV")

useful_path = "data/DC/useful.CSV"
useless_path = "data/DC/useless.CSV"
stop_words_path = "data/chinese_stop_words.txt"

# process data frame
patent = PatentDoc(useful_path, useless_path, stop_words_path)
useful = patent.useful_docs
useless = patent.useless_docs

useful_abs = (useful.摘要 + useful.第一权利要求).tolist()
useless_abs = (useless.摘要 + useless.第一权利要求).tolist()

useful_abs_cut = patent.cut_words(useful_abs, 1)
useless_abs_cut = patent.cut_words(useless_abs, 1)

all_docs = useful_abs_cut + useless_abs_cut
label = np.concatenate([np.repeat(1, len(useful)), 
                        np.repeat(0, len(useless))])

doc_bow, doc_tfidf, doc_dict, _ = patent.bow_tfidf_matrix(all_docs)

#------------------predict by bow ---------------
train_x, test_x, train_y, test_y = train_test_split(doc_bow, 
                                                    label,
                                                    stratify = label,
                                                    test_size = 0.3, random_state = 88)
# using smote
smote_train_x, smote_train_y = SVMSMOTE().fit_resample(train_x, train_y)
bow_mdl = PatentModel(smote_train_x, smote_train_y, test_x, test_y)
bow_lr_mdl, bow_lr_test_report = bow_mdl.logistic_reg()
# bow_nb_mdl, bow_nb_test_report = bow_mdl.naive_bayes()
bow_svm_mdl, bow_svm_test_report = bow_mdl.svm()
bow_nn_mdl, bow_nn_test_report = bow_mdl.neural_network()

#----------------predict by tfidf------------------
train_x, test_x, train_y, test_y = train_test_split(doc_tfidf, label, test_size = 0.3, random_state = 88)
tfidf_mdl = PatentModel(train_x, train_y, test_x, test_y, "tfidf")
tfidf_lr_mdl, tfidf_lr_test_report = tfidf_mdl.logistic_reg()
# tfidf_nb_mdl, tfidf_nb_test_report = tfidf_mdl.naive_bayes()
tfidf_svm_mdl, tfidf_svm_test_report = tfidf_mdl.svm()
tfidf_nn_mdl, tfidf_nn_test_report = tfidf_mdl.neural_network()

# plot roc curve
mdls = [bow_lr_mdl,  bow_svm_mdl, bow_nn_mdl]
mdl_names = ["LR",  "SVM", "NN"]
bow_mdl.roc_curve_plot(mdls, mdl_names, "Bow ROC Curves")
pyplot.show()

# tfidf roc curve
mdls = [tfidf_lr_mdl, tfidf_svm_mdl, tfidf_nn_mdl]
mdl_names = ["LR", "SVM", "NN"]
tfidf_mdl.roc_curve_plot(mdls, mdl_names, "TFIDF ROC Curves")
pyplot.show()

