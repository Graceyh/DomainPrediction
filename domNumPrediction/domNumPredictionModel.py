# build model for domain number prediction
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.grid_search  import GridSearchCV
from sklearn import metrics
import xgboost as xgb
# plotting
import matplotlib.pyplot as plt
# %matplotlib inline

import seaborn as sns

# read Ab original data
Ab_data = pd.read_csv("./AbData(allfeature).csv")

# drop the identifier', 'sequence' column of Ab_data to form the dataframe to build model
Ab_data.drop(Ab_data.columns[[0,2]],axis=1,inplace=True)

#-----------------------------------------------------------------------------#
# choose 11 features with top F scores
xgb_features = ['num_dom', 'avg_buriedProportion', 'length','perc_C', 'avg_hydrophobicity','avg_Flexibility','perc_H','perc_V','perc_W','perc_S', 'solvationFreeEnergy', 'perc_E']

# choose features with importance by RF greater than 0.04
RF_features1 = ['num_dom', 'length', 'avg_hydrophobicity', 'avg_Flexibility', 'avg_buriedProportion']
# choose features with importance by RF greater than 0.04
RF_features2 = ['num_dom', 'length','avg_hydrophobicity', 'avg_Flexibility', 'avg_solvAccess', 'avg_buriedProportion','avg_buriedRatio']
# delete some redundant features based on correlation of avg_hydrophobicity and other features
RF_features3 = ['num_dom', 'length','avg_hydrophobicity', 'avg_solvAccess']

# delete some redundant features based on correlation of avg_Flexibility and other features
RF_features4 = ['num_dom', 'length','avg_Flexibility', 'avg_solvAccess']
# delete some redundant features based on correlation of avg_BuriedProportion and other features
RF_features5 = ['num_dom', 'length','avg_solvAccess','avg_buriedProportion']

# delete some redundant features based on correlation of avg_BuriedRatio and other features
RF_features6 = ['num_dom', 'length','avg_solvAccess','avg_buriedRatio']

# #-----------------------------------------------------------------------------#
# create dataset with only features in xgb_feature set
# Ab_data = Ab_data[xgb_features]
# create dataset with only features in RF_feature1 set
# Ab_data = Ab_data[RF_features1]

#-----------------------------------------------------------------------------#
# draw correlation map
# corr = Ab_data_AA.corr()
# # generate a customed diverging colormap
# cmap = sns.diverging_palette(220,10,as_cmap = True)
# # draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr,vmax = 1, vmin = -1)
# plt.title('correlation heatmap of frequencies of 20 amino acids')
# plt.ylabel('amino acid')
# plt.xlabel('amino acid')
# plt.show()

# print(corr)

#-----------------------------------------------------------------------------#
# model on Ab data
# prepare the imputs of all the features
y_Ab = Ab_data['num_dom']
# print(y_Ab.value_counts())
X_Ab = Ab_data.copy().drop(['num_dom'],axis=1)

# 2.2 randomly split the Ab dataset into training set and test set with 90%:10% ratio
X_Ab_train,X_Ab_test,y_Ab_train, y_Ab_test = cross_validation.train_test_split(X_Ab,y_Ab, test_size=0.1, random_state=0)

#---------------------------------------------------------------
# 1. train models without cross validation and test on 10% test set
#---------------------------------------------------------------
# 1.1 train the linear model on the training data
LR_clf_Ab = linear_model.Lasso(alpha = 0.1)
LR_clf_Ab.fit(X_Ab_train,y_Ab_train)

# predict accuracy
LR_acc_Ab= LR_clf_Ab.score(X_Ab_test,y_Ab_test)
print("Accuracy of linear regression: {:.4f}".format(LR_acc_Ab))

#-----------------------------------------------------------------------------#
# 1.2 train the logistic regression model on the training data
Logistic_clf_Ab = linear_model.LogisticRegression()
Logistic_clf_Ab.fit(X_Ab_train,y_Ab_train)

# predict accuracy
Logistic_acc_Ab= Logistic_clf_Ab.score(X_Ab_test,y_Ab_test)

print("Accuracy of logistic regression: {:.4f}".format(Logistic_acc_Ab))

#-----------------------------------------------------------------------------#
# 1.3 train the SGDclassifier on the training data
SGD_clf_Ab = linear_model.SGDClassifier()
SGD_clf_Ab.fit(X_Ab_train,y_Ab_train)

# predict accuracy
SGD_acc_Ab= SGD_clf_Ab.score(X_Ab_test,y_Ab_test)
print("Accuracy of SGD: {:.4f}".format(SGD_acc_Ab))
#-----------------------------------------------------------------------------#
# 1.4 train the SVM model on the training data
SVM_clf_Ab = svm.SVC(decision_function_shape='ovo')
SVM_clf_Ab.fit(X_Ab_train,y_Ab_train)

# predict accuracy
SVM_acc_Ab= SVM_clf_Ab.score(X_Ab_test,y_Ab_test)
print("Accuracy of SVM: {:.4f}".format(SVM_acc_Ab))

#------------------------------
# 1.5 train the SVC model on the training data
linSVC_clf_Ab = svm.LinearSVC()
linSVC_clf_Ab.fit(X_Ab_train,y_Ab_train)

# predict accuracy
linSVC_acc_Ab= linSVC_clf_Ab.score(X_Ab_test,y_Ab_test)
print("Accuracy of linear SVC regression: {:.4f}".format(linSVC_acc_Ab))

#-----------------------------------------------------------------------------#
# 1.6 train the random forest classifier on the training data
RF_clf_Ab = RandomForestClassifier(n_estimators=10)
RF_clf_Ab.fit(X_Ab_train,y_Ab_train)

# predict accuracy
RF_acc_Ab= RF_clf_Ab.score(X_Ab_test,y_Ab_test)
print("Accuracy of RF: {:.4f}".format(RF_acc_Ab))

# draw feature importance
plt.plot(RF_clf_Ab.feature_importances_)
plt.title("Feature importance by Random Forest Model")
plt.show()
# #-----------------------------------------------------------------------------#
# 1,7 train the XGBclassifier on the training data
xgb_clf_Ab = xgb.XGBClassifier()
xgb_clf_Ab.fit(X_Ab_train,y_Ab_train)

# predict accuracy
xgb_acc_Ab= xgb_clf_Ab.score(X_Ab_test,y_Ab_test)
print("Accuracy of xgb: {:.4f}".format(xgb_acc_Ab))

# draw feature importance
xgb.plot_importance(xgb_clf_Ab,title = 'Feature importance',xlabel = 'F score', ylabel = 'Features', grid = True )
plt.show()

#-----------------------------------------------------------------------------#
# 2. train the models on the training data with 5-fold cross validation and check the stability of prediction accuracy
#-----------------------------------------------------------------------------#
# 2.1 train the logistic regression model on the training data with 5-fold cross validation
LR_cv = linear_model.LogisticRegression()
LR_scores = cross_validation.cross_val_score(LR_cv, X_Ab,y_Ab, cv = 5)
print("Logistic Regression")
print(LR_scores)
print("Accuracy of LR: %0.2f (+/- %0.2f)" %(LR_scores.mean(),LR_scores.std()*2))

#-----------------------------------------------------------------------------#
# 2.2 train SVM classifier with 5-fold cross validation

SVM_cv = svm.SVC(decision_function_shape='ovo',class_weight = 'balanced')
# predict accuracy
SVM_scores = cross_validation.cross_val_score(SVM_cv, X_Ab,y_Ab, cv = 5)
print("SVM")
print(SVM_scores)
print("Accuracy of svm: %0.2f (+/- %0.2f)" %(SVM_scores.mean(),SVM_scores.std()*2))

#-----------------------------------------------------------------------------#
# 2.3 train RF classifier with five-fold cross validation
RF_cv = RandomForestClassifier()
# predict accuracy
RF_scores = cross_validation.cross_val_score(RF_cv, X_Ab,y_Ab, cv = 5)

print("Random Forest")
print(RF_scores)
print("Accuracy of rf: %0.2f (+/- %0.2f)" %(RF_scores.mean(),RF_scores.std()*2))

#-----------------------------------------------------------------------------#
# 2.4 train XGB classifier with five-fold cross validation
xgb_cv = xgb.XGBClassifier()
# predict accuracy
xgb_scores = cross_validation.cross_val_score(xgb_cv, X_Ab,y_Ab, cv = 5)

print("XGboost")
print(xgb_scores)
print("Accuracy of xgb: %0.2f (+/- %0.2f)" %(xgb_scores.mean(),xgb_scores.std()*2))


#-----------------------------------------------------------------------------#
# 2. Grid search and find optimal parameter for logistic regression, SVM and random forest, XGBoost
#-----------------------------------------------------------------------------#
LRclf_cv = linear_model.LogisticRegression()

LR_parameters = {
    'penalty': ['l1','l2']
}

LRclf_cv = GridSearchCV(LRclf_cv, LR_parameters, n_jobs = 1, cv = 3)

LRclf_cv.fit(X_Ab,y_Ab)

# gridScoreLR = LRclf_cv.grid_scores_
# print(gridScoreLR)

LRbest_estimator = LRclf_cv.best_estimator_
LRbest_score = LRclf_cv.best_score_
LRbest_params = LRclf_cv.best_params_

print("-------------------------------------------------------")
print(LRbest_estimator)
print("the best accuracy of LR is: " + str(LRbest_score))
print("the optimal paramters of LR is: " + str(LRbest_params))

#-----------------------------------------------------------------------------#
# 4.2 train SVM classifier with 3-fold cross validation

SVMclf_cv = svm.SVC(decision_function_shape='ovo', class_weight = 'balanced')
parameters = {
    'kernel': ['rbf', 'linear'],
}

SVMclf_cv = GridSearchCV(SVMclf_cv, parameters, cv = 3)

SVMclf_cv.fit(X_Ab,y_Ab)

gridScoreSVM = SVMclf_cv.grid_scores_

y_casp_pred_SVM = SVMclf_cv.predict(X_CA)
SVM_score_casp = metrics.accuracy_score(y_CA, y_casp_pred_SVM)
print("the prediction accuracy of SVM on CASP data is: " + str(SVM_score_casp))

print(gridScoreSVM)

SVMbest_estimator = SVMclf_cv.best_estimator_
SVMbest_score = SVMclf_cv.best_score_
SVMbest_params = SVMclf_cv.best_params_
print(gridScoreSVM)
print("-------------------------------------------------------")
print(SVMbest_estimator)
print("the best accuracy of SVM is: " + str(SVMbest_score))
print("the optimal paramters of SVM is: " + str(SVMbest_params))
print("the prediction accuracy of SVM on CASP data is: " + str(SVM_score_casp))

#-----------------------------------------------------------------------------#
# 5.2 train RF classifier with three-fold cross validation
max_features = int(np.log2(len(X_Ab.columns)))+1
RFclf_cv = RandomForestClassifier(max_features = max_features, class_weight = 'balanced')
RF_parameters = {
    'n_estimators': [200,250,300,400],
}

RFclf_cv = GridSearchCV(RFclf_cv, RF_parameters, n_jobs = 1, cv = 3)

RFclf_cv.fit(X_Ab,y_Ab)

# gridScoreRF = RFclf_cv.grid_scores_
# print(gridScoreRF)

RFbest_estimator = RFclf_cv.best_estimator_
RFbest_score = RFclf_cv.best_score_
RFbest_params = RFclf_cv.best_params_

print("-------------------------------------------------------")
print(RFbest_estimator)
print("the best accuracy of RF is: " + str(RFbest_score))
print("the optimal paramters of RF is: " + str(RFbest_params))

#-----------------------------------------------------------------------------#
# 5.2 train XGB classifier with 3-fold cross validation
xgb_clf_cv = xgb.XGBClassifier()
# predict accuracy
xgb_parameters = {
    # 'max_depth': [10,20],
    'max_depth': [20, 30],
    'learning_rate': [0.05,0.1]
}

xgb_clf_cv = GridSearchCV(xgb_clf_cv, xgb_parameters, n_jobs = 1, cv = 3)

xgb_clf_cv.fit(X_Ab,y_Ab)

xgb_best_estimator = xgb_clf_cv.best_estimator_
xgb_best_score = xgb_clf_cv.best_score_
xgb_best_params = xgb_clf_cv.best_params_

print("-------------------------------------------------------")
print("Xgboost")
print(xgb_best_estimator)
print("the best accuracy of xgb is: " + str(xgb_best_score))
print("the optimal paramters of xgb is: " + str(xgb_best_params))
