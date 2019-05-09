import pandas as pd
import numpy as np
import re
import jieba


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
    
  def cut_words(self, docs):
    
    word_docs = []
    for text in docs:
      text = re.sub("[^\w\s]", "", text)
      text = re.sub("\d+", "", text)
      text = re.sub("\s+", "", text)
      text = re.sub("[a-z]+", "", text)
      word_docs.append([word for word in jieba.cut(text) if word not in self.stop_words])
      
    return word_docs
  
  @staticmethod
  def doc_vector(text, word2vec_mdl):
    
    word_mat = [word2vec_mdl[word] for word in text if word in word2vec_mdl]
    text_score = np.array(word_mat).mean(axis = 0)
    return text_score
    
  @classmethod
  def text_score_vector(cls, docs, word2vec_mdl):
    
    return [cls.doc_vector(text, word2vec_mdl) for text in docs]
