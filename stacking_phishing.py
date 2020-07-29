import pandas as pd
import numpy as np
from numpy import sort
from pandas.tools.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from vecstack import stacking
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#from mlxtend.classifier import EnsembleVoteClassifier,StackingCVClassifier,StackingClassifier
# Going to use these 5 base models for the stacking
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,roc_curve,brier_score_loss,precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
#from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.linear_model import SGDClassifier


from sklearn.naive_bayes import GaussianNB



url = "dataset/phising_2456.csv"
names = ["has_ip", "long_url", "short_service", "has_at", "double_slash_redirect",
           "pref_suf", "has_sub_domain", "ssl_state", "long_domain", "favicon", "port",
           "https_token", "req_url", "url_of_anchor", "tag_links", "SFH", 
           "submit_to_email", "abnormal_url", "redirect", "mouseover", "right_click",
           "popup", "iframe", "domain_Age", "dns_record", "traffic", "page_rank",
           "google_index", "links_to_page", "stats_report", "target"]

dataset = pd.read_csv(url)

	
# # shape
# print(dataset.shape)

# # head
# #print(dataset.head(20))

# # descriptions
# print(dataset.describe())

# class distribution
#print(dataset.groupby('target').size())

# box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2,31), sharex=False, sharey=False)
# plt.show()

# histograms
# dataset.hist()
# plt.show()

# scatter plot matrix
# scatter_matrix(dataset)
# plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:30]
Y = array[:,30]
validation_size = 0.35
seed = 7
X_train, X_test, y_train, y_test= model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
# seed = 7
# scoring = 'accuracy'


# # Spot Check Algorithms
# models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))
# # evaluate each model in turn
# results = []
# names = []
# for name, model in models:
# 	kfold = model_selection.KFold(n_splits=10, random_state=seed)
# 	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
# 	results.append(cv_results)
# 	names.append(name)
# 	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
# 	print(msg)

# #Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# build model stack
# stack = StackingClassifier(classifiers = models)
# # kfold = model_selection.KFold(n_splits=10, random_state=seed)
# # cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
# # results.append(cv_results)
# # names.append(name)
# # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
# # print(msg)


# stack.fit(X_train, Y_train)

# # make predictions for our test set
# ydata_test_pred = stack.predict_proba(X_validation)[:,1]

# # determine cutoff balancing precision/recall
# precision, recall, threshold = precision_recall_curve(Y_validation, ydata_test_pred)
# pos_threshold = np.min(threshold[precision[1:] > recall[:-1]])
# print('Positive threshold: %s' % str(pos_threshold))
# print('Confusion matrix:')
# print(confusion_matrix(Y_validation, (ydata_test_pred >= pos_threshold).astype(int)))
# print('Stack AUC: %s' % roc_auc_score(Y_validation, ydata_test_pred))
#meta_classifier = RandomForestClassifier();
# stack = StackingClassifier(models)
# stack.fit(X_train,Y_train)
# predictions = stack.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))

# clf1 = LogisticRegression()
# clf2 = LinearDiscriminantAnalysis()
# clf3 = KNeighborsClassifier()
# clf4 = DecisionTreeClassifier()
# clf5 = GaussianNB()
# clf6 = SVC()

# rf = RandomForestClassifier()

# sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4,  clf6], 
#                           meta_classifier=rf)

# print('3-fold cross validation:\n')

# for clf, label in zip([clf1, clf2, clf3, clf4,  clf6, sclf], 
#                       ['LR', 
#                        'LDA', 
#                        'KNN',
#                        'CART',
#                        'SVM',
#                        'StackingClassifier']):

#     scores = model_selection.cross_val_score(clf, X_train, Y_train, 
#                                               cv=10, scoring='accuracy')
#     print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
#           % (scores.mean(), scores.std(), label))

# Make predictions on validation dataset
# knn = KNeighborsClassifier()
# knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))


#pipe1 = make_pipeline(ColumnSelector(cols=(0, 2)),
#                      ExtraTreesClassifier())
#pipe2 = make_pipeline(ColumnSelector(cols=(1, 2, 3)),
#                      ExtraTreesClassifier())
#
#sclf = StackingClassifier(classifiers=[pipe1, pipe2], 
#                          meta_classifier=ExtraTreesClassifier())
#
#sclf.fit(X_train, Y_train)
#
#predictions = sclf.predict(X_validation)
#
#print(accuracy_score(Y_validation, predictions))

# clf = RandomForestClassifier()
# clf.fit(X, Y)
# importances = clf.feature_importances_
# 	#std = np.std([tree.feature_importances_ for tree in forest.estimators_],
# 	 #           axis=0)
# indices = np.argsort(importances)[::-1]


# # Print the feature ranking
# print("Feature ranking:")

# for f in range(X.shape[1]):
# 	print("%d. feature %d (%f)" % (f + 1, indices[f]+1, importances[indices[f]]))
# 	    #indice.append(f + 1)
# # 	    features.append(indices[f]+1)
# # 	    rank.append(importances[indices[f]])
# thresholds = sort(clf.feature_importances_)
# i = 0
# Threshes = []
# nth = []
# acu = []
# fpr = []
# tpr = []
# index = []
# new = []
# # for i in range(100):
# for thresh in thresholds:
#     # select features using threshold
#     selection = SelectFromModel(clf, threshold=thresh, prefit=True)
#     select_X_train = selection.transform(X_train)
#     # train model
#     selection_model = RandomForestClassifier()
#     selection_model.fit(select_X_train, y_train)
#     # eval model
#     select_X_test = selection.transform(X_test)
#     y_pred = selection_model.predict(select_X_test)

#     predictions = [round(value) for value in y_pred]
#     accuracy = accuracy_score(y_test, predictions)

#     # y_pred_dt = selection_model.predict_proba(select_X_test)[:, 1]
#     # fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)
#     # index.append(i)
#     # fpr.append(fpr_dt)
#     # tpr.append(tpr_dt)
#     # new.append(select_X_train.shape[1])
#     # #print(fpr_dt,tpr_dt)
#     # precision_dt, recall_dt, _ = precision_recall_curve(y_test, y_pred_dt)
#     i= i+1

#         # et = ExtraTreesClassifier()
#         # et.fit(x_train, y_train)
#         # y_pred_et = et.predict_proba(x_test)[:, 1]
#         # fpr_et, tpr_et, _ = roc_curve(y_test, y_pred_et)
#         # precision_et, recall_et, _ = precision_recall_curve(y_test, y_pred_et)


#         # rf = RandomForestClassifier()
#         # rf.fit(x_train, y_train)
#         # y_pred_rf = rf.predict_proba(x_test)[:, 1]
#         # fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
#         # precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_rf)


#         # ada = AdaBoostClassifier()
#         # ada.fit(x_train, y_train)
#         # y_pred_ada = ada.predict_proba(x_test)[:, 1]
#         # fpr_ada, tpr_ada, _ = roc_curve(y_test, y_pred_ada)
#         # precision_ada, recall_ada, _ = precision_recall_curve(y_test, y_pred_ada)


#         # grd = GradientBoostingClassifier()
#         # grd.fit(x_train, y_train)
#         # y_pred_grd = grd.predict_proba(x_test)[:, 1]
#         # fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)
#         # precision_grd, recall_grd, _ = precision_recall_curve(y_test, y_pred_grd)
    
#     # print(classification_report(y_test, predictions))
#     print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
#         # Threshes.append(thresh)
#         # nth.append(select_X_train.shape[1])
#         # acu.append(accuracy)
# #         i= i+1

# plt.figure(1)
# plt.xlim(0,1)
# plt.ylim(0, 1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr[7], tpr[7], label='%d' %new[7])
# plt.plot(fpr[6], tpr[6], label='%d' %new[6])
# plt.plot(fpr[5], tpr[5], label='%d' %new[5])
# plt.plot(fpr[4], tpr[4], label='%d' %new[4])
# plt.plot(fpr[3], tpr[3], label='%d' %new[3])
# plt.plot(fpr[2], tpr[2], label='%d' %new[2])
# plt.plot(fpr[1], tpr[1], label='%d' %new[1])
# plt.plot(fpr[0], tpr[0], label='%d' %new[0])
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC Curve')
# plt.legend(loc='best')
# plt.show()	


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)
precision_dt, recall_dt, _ = precision_recall_curve(y_test, y_pred_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

mlp = MLPClassifier()
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict_proba(X_test)[:, 1]
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_pred_mlp)
precision_mlp, recall_mlp, _ = precision_recall_curve(y_test, y_pred_mlp)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

svm = svm.SVC(probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict_proba(X_test)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm)
precision_svm, recall_svm, _ = precision_recall_curve(y_test, y_pred_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

sgd = SGDClassifier(loss='log')
sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict_proba(X_test)[:, 1]
fpr_sgd, tpr_sgd, _ = roc_curve(y_test, y_pred_sgd)
precision_sgd, recall_sgd, _ = precision_recall_curve(y_test, y_pred_sgd)
roc_auc_sgd = auc(fpr_sgd, tpr_sgd)

gb = GaussianNB()
gb.fit(X_train, y_train)
y_pred_gb = gb.predict_proba(X_test)[:, 1]
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_gb)
precision_gb, recall_gb, _ = precision_recall_curve(y_test, y_pred_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)





models = [
            
    RandomForestClassifier(),
        
    DecisionTreeClassifier(),
    
    MLPClassifier(),
  
    SVC(),

    SGDClassifier(),

    GaussianNB()]
    

# Compute stacking features
S_train, S_test = stacking(models, X_train, y_train, X_test, 
    regression = False, metric = accuracy_score, n_folds = 10, 
    stratified = True, shuffle = True, random_state = 0, verbose = 2)

# Initialize 2nd level model
model = XGBClassifier()
    
# Fit 2nd level model
model = model.fit(S_train, y_train)

# Predict
y_pred = model.predict(S_test)

y_pred_model = model.predict_proba(S_test)[:, 1]
fpr_model, tpr_model, _ = roc_curve(y_test, y_pred_model)
precision_model, recall_model, _ = precision_recall_curve(y_test, y_pred_model)
roc_auc_model = auc(fpr_model, tpr_model)

plt.figure()

plt.plot(recall_rf, precision_rf, color='darkorange',
         lw=2, label='Random Forest',marker=".")

plt.plot(recall_dt, precision_dt, color='darkgreen',
         lw=2, label='Decision Tree',marker=".")

plt.plot(recall_mlp, precision_mlp, color='darkblue',
         lw=2, label='Multi Layer Perception',marker=".")

plt.plot(recall_svm, precision_svm, color='darkslateblue',
         lw=2, label='Support Vector Machine',marker=".")

plt.plot(recall_sgd, precision_sgd, color='darkolivegreen',
         lw=2, label='Stochastic Gradient Descent',marker=".")

plt.plot(recall_gb, precision_gb, color='indigo',
         lw=2, label='Gaussian Naive Bayes',marker=".")

plt.plot(recall_model, precision_model, color='red',
         lw=2, label='Stacked Generalization',marker=".")

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision_Recall Curve')
plt.legend(loc="lower right")
plt.show()
# Final prediction score
print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pred))
print('Final Absulate Error score: [%.8f]' % mean_absolute_error(y_test, y_pred))

