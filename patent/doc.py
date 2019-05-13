import pandas as pd
import numpy as np
import re
import jieba
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter


class PatentDoc:
  
  def __init__(self, useful_path, useless_path, stop_words_path):
    
    self._useful_path = useful_path
    self._useless_path = useless_path
    self._stop_words_path = stop_words_path
    self.useful_docs = self._read_useful()
    self.useless_docs = self._read_useless()
    self.stop_words = self._read_stop_words()
    
  
  def _read_useful(self):
    
    return pd.read_csv(self._useful_path)
    
  def _read_useless(self):
    
    return pd.read_csv(self._useless_path)
    
  def _read_stop_words(self):
    
    with open(self._stop_words_path, encoding = "utf-8") as handle:
      words = handle.readlines()
      
    return [word.strip() for word in words if word.strip()]
    
  def cut_words(self, docs, filter_len = 0):
    
    word_docs = []
    for text in docs:
      text = re.sub("[^\w\s]", "", text)
      text = re.sub("\d+", "", text)
      text = re.sub("\s+", "", text)
      text = re.sub("[a-z]+", "", text)
      word_docs.append([word for word in jieba.cut(text) 
                        if word not in self.stop_words
                        and len(word) > filter_len])
      
    return word_docs
    
  @staticmethod
  def bow_tfidf_matrix(docs, smooth_idf = True):
    mat_dict = DictVectorizer()
    mat = mat_dict.fit_transform(Counter(doc) for doc in docs)
    
    tfidf = TfidfTransformer(smooth_idf = smooth_idf)
    tfidf_mat = tfidf.fit_transform(mat.toarray())
    
    return mat, tfidf_mat, mat_dict, tfidf
  
  @staticmethod
  def doc_vector(text, word2vec_mdl):
    
    word_mat = [word2vec_mdl[word] for word in text if word in word2vec_mdl]
    text_score = np.array(word_mat).mean(axis = 0)
    return text_score
    
  @classmethod
  def text_score_vector(cls, docs, word2vec_mdl):
    
    return [cls.doc_vector(text, word2vec_mdl) for text in docs]
    
  @staticmethod  
  def transform_predict(bow_processor, tf_idf_processor, new_data):
    
    bow_mat = bow_processor.transform(Counter(doc) for doc in new_data)
    tf_idf_mat = tf_idf_processor.transform(bow_mat.toarray())
    
    return bow_mat, tf_idf_mat
    
