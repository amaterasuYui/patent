from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from matplotlib import pyplot


class PatentModel:
  
  def __init__(self, train_x, train_y, test_x, test_y, data_descr = "bow"):
    
    self.train_x = train_x
    self.train_y = train_y
    self.test_x = test_x
    self.test_y = test_y
    self.data_description = data_descr

    
  def logistic_reg(self):
    
    lr_mdl = LogisticRegression()
    lr_mdl.fit(self.train_x, self.train_y)
    test_report = classification_report(self.test_y, lr_mdl.predict(self.test_x))
    
    return lr_mdl, test_report
    
  def naive_bayes(self):
    mdl = MultinomialNB()
    mdl.fit(self.train_x, self.train_y)
    test_report = classification_report(self.test_y, mdl.predict(self.test_x))
    
    return mdl, test_report
    
  def svm(self, random_state = 24):
    
    svm = SGDClassifier(random_state = random_state)
    mdl = CalibratedClassifierCV(svm)
    mdl.fit(self.train_x, self.train_y)
    test_report = classification_report(self.test_y, mdl.predict(self.test_x))
    
    return mdl, test_report
  
  def neural_network(self,  
                     activation = ["relu", "relu", "sigmoid"],
                     n_nodes = [6, 3, 1],
                     epochs = 20,
                     loss_func = "binary_crossentropy",
                     metrics = ["accuracy"],
                     lr_rate = 0.05,
                     batch_size = 32,
                     validation_split = 0.1,
                     verbose = 1):
    if len(activation) != len(n_nodes):
      raise Exception("activation list lenght should be the same as n_layer")
    
    model = Sequential()
    layer_index = 1
    for nodes, activ in zip(n_nodes, activation):
      
      if layer_index == 1:
        model.add(Dense(nodes, input_shape = (self.train_x.shape[1],), activation = activ))
      else:
        model.add(Dense(nodes, activation = activ))
      layer_index += 1
    
    model.compile(Adam(lr = lr_rate), loss_func, metrics = metrics)
    model.fit(self.train_x, self.train_y, 
              batch_size = batch_size, 
              epochs = epochs, 
              validation_split = validation_split, 
              verbose = verbose)
    test_report = classification_report(self.test_y, model.predict_classes(self.test_x))
    
    return model, test_report
  
  def roc_curve_point(self, mdl):
    auc = []
    fpr = []
    tpr = []
    
    for model in mdl:
      test_prob = model.predict_proba(self.test_x)
      
      if "keras" in str(type(model)) :
        test_prob = test_prob.ravel()
      else:
        test_prob = test_prob[:, 1]
      auc.append(roc_auc_score(self.test_y, test_prob))
      mdl_fpr, mdl_tpr, _ = roc_curve(self.test_y, test_prob)
      fpr.append(mdl_fpr)
      tpr.append(mdl_tpr)
    
    return [auc, fpr, tpr]
    
  def roc_curve_plot(self, mdl, mdl_names, title):
    roc_metrics = self.roc_curve_point(mdl)
    for auc, fpr, tpr, mdl_name in zip(*roc_metrics, mdl_names):
      pyplot.plot([0, 1], [0, 1], linestyle = "--")
      pyplot.plot(fpr, tpr, marker = "", label = "%s ROC curve (area = %0.2f)" % (mdl_name, auc))
    
    pyplot.title(title)
    pyplot.legend(loc = "lower right")
