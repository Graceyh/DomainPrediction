# train models on specified dataset with or without cross validation and perform parameter tuning on SVM, Random Forest
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.grid_search  import GridSearchCV

import xgboost as xgb
# plotting
import matplotlib.pyplot as plt
# %matplotlib inline

import seaborn as sns

# read chunk data
# Perc_chunk(allfeature1).csv: positive/ negative = 8000/9650
# chunk_data = pd.read_csv("/Users/Graceyh/Desktop/ProcessedData/Perc_chunk(allfeature1).csv")
# chunk_data = pd.read_csv("/Users/Graceyh/Desktop/ProcessedData/Perc_chunk(allfeature+linkIdx).csv")
chunk_data = pd.read_csv("/Users/Graceyh/Desktop/ProcessedData/Perc_chunk(allfeatureLen13).csv")

print(len(chunk_data[chunk_data['dbInd']==1]))
print(len(chunk_data[chunk_data['dbInd']==0]))
print(chunk_data.columns)
# drop the 'AA_column' column of chunk_data to form the dataframe to build model
chunk_data.drop(chunk_data.columns[0],axis=1,inplace=True)

# shuffle the dataset
chunk_data = chunk_data.sample(frac = 1)
# print(chunk_data.columns)

#-----------------------------------------------------------------------------#
# choose 11 features with top F scores
xgb_features1 = ['dbInd', 'sum_volume','avg_hydrophobicity','solvationFreeEnergy', 'avg_Flexibility', 'avg_solvAccess', 'avg_buriedProportion', 'entropy','sum_size','perc_P','perc_L', 'perc_K']
# choose 11 features with top F scores
xgb_features2 = ['dbInd', 'avg_linkeIdx','sum_volume', 'avg_hydrophobicity','solvationFreeEnergy','avg_Flexibility', 'avg_buriedProportion', 'sum_size','perc_P', 'perc_S','perc_N','perc_W']
# # choose features with importance by RF greater than or equal to 0.03
RF_features1 = ['dbInd', 'avg_hydrophobicity', 'sum_size','sum_volume', 'avg_Flexibility', 'avg_solvAccess', 'solvationFreeEnergy', 'avg_linkeIdx']
# choose features with importance by RF greater than or equal to 0.03
RF_features2 = ['dbInd', 'perc_P', 'avg_hydrophobicity', 'sum_size', 'sum_volume', 'sum_netCharge', 'avg_Flexibility', 'avg_solvAccess', 'avg_buriedProportion', 'solvationFreeEnergy', 'entropy']

# use dataset of xgb_features
# chunk_data = chunk_data[xgb_features2]

# use dataset of RFfeatures
# chunk_data = chunk_data[RF_features1]
#-----------------------------------------------------------------------------#
# compute mutual information of the explanatory variables and the explained variable
MI_all = []
adj_MI_all = []
for column in chunk_data.columns:
    # MI_all.append(metrics.mutual_info_score(chunk_data[column],chunk_data['dbInd']))
    if column == 'dbInd':
        pass
    else:
        adj_MI_all.append(metrics.adjusted_mutual_info_score(chunk_data[column],chunk_data['dbInd']))
print(chunk_data.columns)
print(adj_MI_all)
#
print("mutual information of " + \
 str(chunk_data.columns[np.argmax(adj_MI_all)]) + \
     " and explained variable is the highest, the value is " +\
      str(max(adj_MI_all)))
#-----------------------------------------------------------------------------#
# 2. model on chunk data
# 2.1 prepare the imputs of all the features for the regression model
y_chunk = chunk_data['dbInd']
print(len(y_chunk))
X_chunk = chunk_data.drop(['dbInd'],axis=1)
print(X_chunk.head())

# 2.2 randomly split the chunk dataset into training set and test set with 90%:10% ratio
X_chunk_train,X_chunk_test,y_chunk_train, y_chunk_test = cross_validation.train_test_split(X_chunk,y_chunk, test_size=0.1, random_state=0)

#-------------------------------------------------------------------
# train models without cross validation and validate on test dataset
#-------------------------------------------------------------------
# 1. train the linear model on the training data
LR_clf_chunk = linear_model.Lasso(alpha = 0.1)
LR_clf_chunk.fit(X_chunk_train,y_chunk_train)
LR_acc_chunk= LR_clf_chunk.score(X_chunk_test,y_chunk_test)
print("Accuracy of linear regression: {:.4f}".format(LR_acc_chunk))

#------------------------------------------------------------
# 2. train the logistic regression model on the training data
Logistic_clf_chunk = linear_model.LogisticRegression()
Logistic_clf_chunk.fit(X_chunk_train,y_chunk_train)

# predict accuracy
Logistic_acc_chunk= Logistic_clf_chunk.score(X_chunk_test,y_chunk_test)
print("Accuracy of logistic regression: {:.4f}".format(Logistic_acc_chunk))

#------------------------------------------------
# 3. train the SGDclassifier on the training data
SGD_clf_chunk = linear_model.SGDClassifier()
SGD_clf_chunk.fit(X_chunk_train,y_chunk_train)

# predict accuracy
SGD_acc_chunk= SGD_clf_chunk.score(X_chunk_test,y_chunk_test)
print("Accuracy of SGD: {:.4f}".format(SGD_acc_chunk))

#---------------------------------------------------
# 4. train SVM classifier on training dataset and evaluate with accuracy, precision, recall and F1, MCC
SVM_clf_chunk = svm.SVC(decision_function_shape='ovo')
SVM_clf_chunk.fit(X_chunk_train,y_chunk_train)
SVM_chunk_pred = SVM_clf_chunk.predict(X_chunk_test)
print(SVM_chunk_pred)

# predict accuracy
SVM_acc_chunk= SVM_clf_chunk.score(X_chunk_test,y_chunk_test)

# compute the precision score on test dataset
SVM_chunk_prec = metrics.precision_score(y_chunk_test,SVM_chunk_pred)

# compute the rec score on test dataset
SVM_chunk_rec = metrics.recall_score(y_chunk_test,SVM_chunk_pred)

# compute the f1 score on test dataset
SVM_chunk_f1 = metrics.f1_score(y_chunk_test,SVM_chunk_pred)

# compute matthews correlation coefficient
SVM_chunk_matt = metrics.matthews_corrcoef(y_chunk_test,SVM_chunk_pred)

print("Accuracy of SVM on test data: {:.4f}".format(SVM_acc_chunk))
print("Precision of SVM on test data: {:.4f}".format(SVM_chunk_prec))
print("Recall of SVM on test data: {:.4f}".format(SVM_chunk_rec))
print("F1 of SVM on test data: {:.4f}".format(SVM_chunk_f1))
print("matthews correlation coefficient of SVM on test data: {:.4f}".format(SVM_chunk_matt))

#--------------------------------#
linSVC_clf_chunk = svm.LinearSVC()
linSVC_clf_chunk.fit(X_chunk_train,y_chunk_train)

# predict accuracy
linSVC_acc_chunk= linSVC_clf_chunk.score(X_chunk_test,y_chunk_test)
print("Accuracy of linear SVC regression: {:.4f}".format(linSVC_acc_chunk))

#-----------------------------------------------------------------------------#
# 5. train the random forest classifier on the training data
RF_clf_chunk = RandomForestClassifier(n_estimators=300)
RF_clf_chunk.fit(X_chunk_train,y_chunk_train)

RF_chunk_pred = RF_clf_chunk.predict(X_chunk_test)
print(RF_chunk_pred)
compute the precision score on test dataset
RF_chunk_prec = metrics.precision_score(y_chunk_test,RF_chunk_pred)

# compute the rec score on test dataset
RF_chunk_rec = metrics.recall_score(y_chunk_test,RF_chunk_pred)

# compute the f1 score on test dataset
RF_chunk_f1 = metrics.f1_score(y_chunk_test,RF_chunk_pred)

# compute the matthews correlation coefficient
RF_chunk_matt = metrics.matthews_corrcoef(y_chunk_test, RF_chunk_pred)

# predict accuracy
RF_acc_chunk= RF_clf_chunk.score(X_chunk_test,y_chunk_test)
print("Accuracy of RF on test data: {:.4f}".format(RF_acc_chunk))
print("Precision of RF on test data: {:.4f}".format(RF_chunk_prec))
print("Recall of RF on test data: {:.4f}".format(RF_chunk_rec))
print("F1 of RF on test data: {:.4f}".format(RF_chunk_f1))
print("matthew correlation coefficient of RF on test data: {:.4f}".format(RF_chunk_matt))

# draw feature importance
plt.plot(RF_clf_chunk.feature_importances_)
plt.title("Feature importance by Random Forest Model")
plt.show()

#-----------------------------------------------------------------------------#
# 6. train the XGBclassifier on the training data
xgb_clf_chunk = xgb.XGBClassifier()
xgb_clf_chunk.fit(X_chunk_train,y_chunk_train)

# predict accuracy
xgb_acc_chunk= xgb_clf_chunk.score(X_chunk_test,y_chunk_test)
print("Accuracy of xgb: {:.4f}".format(xgb_acc_chunk))

# draw feature importance
xgb.plot_importance(xgb_clf_chunk,title = 'Feature importance',xlabel = 'F score', ylabel = 'Features', grid = True )
plt.show()


#------------------------------------------------------------------------------
# find optimal parameters for SVM and Random Forest by 10-fold cross validation
#------------------------------------------------------------------------------
# 4.3 train SVM classifier with 5-fold cross validation and grid research
SVMclf_cv = svm.SVC(class_weight = 'balanced')
parameters = {
    'kernel': ['rbf', 'linear'],
}

SVMclf_cv = GridSearchCV(SVMclf_cv, parameters, cv = 10)

SVMclf_cv.fit(X_chunk,y_chunk)

gridScoreSVM = SVMclf_cv.grid_scores_

print(gridScoreSVM)

SVMbest_estimator = SVMclf_cv.best_estimator_
SVMbest_score = SVMclf_cv.best_score_
SVMbest_params = SVMclf_cv.best_params_
print(gridScoreSVM)
print("-------------------------------------------------------")
print(SVMbest_estimator)
print("the best accuracy of SVM is: " + str(SVMbest_score))
print("the optimal paramters of SVM is: " + str(SVMbest_params))

#----------------------------------------------------#
# 5.3 train RF classifier with 5-fold cross validation and grid search
RFclf_cv = RandomForestClassifier(class_weight = 'balanced')
parameters = {
    'n_estimators': [250,300],
}

RFclf_cv = GridSearchCV(RFclf_cv, parameters, n_jobs = 1, cv = 10)

RFclf_cv.fit(X_chunk,y_chunk)

gridScoreRF = RFclf_cv.grid_scores_
print(gridScoreRF)

RFbest_estimator = RFclf_cv.best_estimator_
RFbest_score = RFclf_cv.best_score_
RFbest_params = RFclf_cv.best_params_

print("-------------------------------------------------------")
print(RFbest_estimator)
print("the best accuracy of RF is: " + str(RFbest_score))
print("the optimal paramters of RF is: " + str(RFbest_params))

#------------------------------------------------------------------------------
# train the models on the training data with 5-fold cross validation and check the stability of prediction accuracy
#------------------------------------------------------------------------------
# 2. train the logistic regression model on the training data with 5-fold cross validation
LR_cv = linear_model.LogisticRegression()
LR_scores = cross_validation.cross_val_score(LR_cv, X_chunk, y_chunk, cv = 5)
print("Logistic Regression")
print(LR_scores)
print("Accuracy: %0.2f (+/- %0.2f)" %(LR_scores.mean(),LR_scores.std()*2))

# ----------------------------------------------------#
# 4.2 train SVM classifier with 5-fold cross validation

SVM_cv = svm.SVC(decision_function_shape='ovo',class_weight = {1:7/4})

# predict accuracy
SVM_pred_cv = cross_validation.cross_val_predict(SVM_cv, X_chunk, y_chunk, cv = 10)
# print("SVM")
# print(SVM_scores)
# print("Accuracy: %0.2f (+/- %0.2f)" %(SVM_scores.mean(),SVM_scores.std()*2))
# SVM_clf_chunk.fit(X_chunk_train,y_chunk_train)
# SVM_chunk_pred = SVM_clf_chunk.predict(X_chunk_test)
# print(SVM_chunk_pred)

# predict accuracy
SVM_acc_chunk=  metrics.accuracy_score(y_chunk,SVM_pred_cv)

# compute the precision score on test dataset
SVM_chunk_prec = metrics.precision_score(y_chunk,SVM_pred_cv)

# compute the rec score on test dataset
SVM_chunk_rec = metrics.recall_score(y_chunk,SVM_pred_cv)

# compute the f1 score on test dataset
SVM_chunk_f1 = metrics.f1_score(y_chunk,SVM_pred_cv)

# compute matthews correlation coefficient
SVM_chunk_matt = metrics.matthews_corrcoef(y_chunk,SVM_pred_cv)

print("Accuracy of SVM with 10-fold cv: {:.4f}".format(SVM_acc_chunk))
print("Precision of SVM with 10-fold cv: {:.4f}".format(SVM_chunk_prec))
print("Recall of SVM with 10-fold cv: {:.4f}".format(SVM_chunk_rec))
print("F1 of SVM with 10-fold cv: {:.4f}".format(SVM_chunk_f1))
print("matthews correlation coefficient of SVM on test data: {:.4f}".format(SVM_chunk_matt))

# ----------------------------------------------------#
# 5.2 train RF classifier with five-fold cross validation
RF_cv = RandomForestClassifier(n_estimators=300)

RF_pred_cv = cross_validation.cross_val_predict(RF_cv, X_chunk, y_chunk, cv = 5)
# predict accuracy
RF_acc_cv = metrics.accuracy_score(y_chunk,RF_pred_cv)

RF_prec_cv = metrics.precision_score(y_chunk,RF_pred_cv)

# compute the rec score on CASP dataset
RF_rec_cv = metrics.recall_score(y_chunk,RF_pred_cv)

# compute the f1 score on CASP dataset
RF_f1_cv = metrics.f1_score(y_chunk,RF_pred_cv)

# compute matthews correlation coefficient
SVM_chunk_matt = metrics.matthews_corrcoef(y_chunk,SVM_pred_cv)

print("Precision of RF with 10-fold cv: {:.4f}".format(RF_prec_cv))
print("Recall of RF with 10-fold cv: {:.4f}".format(RF_rec_cv))
print("F1 score of RF with 10-fold cv: {:.4f}".format(RF_f1_cv))
print("Mean Accuracy of RF with 10-fold cv: {:.4f}".format(RF_acc_cv))
print("matthew correlation of RF with 10-fold cv: {:.4f}".format(RF_acc_cv))
