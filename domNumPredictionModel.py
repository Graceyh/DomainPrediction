import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

import xgboost as xgb
# plotting
import matplotlib.pyplot as plt
# %matplotlib inline

import seaborn as sns

# read Ab original data
Ab_data = pd.read_csv("/Users/Graceyh/Desktop/AbData(allfeature)).csv")
print(Ab_data.columns)
Ab_data_AA = pd.read_csv("/Users/Graceyh/Google Drive/AboutDissertation/Data/ProteinAAperc_correct.csv")
# drop the identifier', 'sequence' column of Ab_data to form the dataframe to build model
Ab_data.drop(Ab_data.columns[[0,2]],axis=1,inplace=True)

print(Ab_data.columns)

#-----------------------------------------------------------------------------#
# # choose 11 features with top F scores
xgb_features = ['num_dom', 'avg_buriedProportion', 'length','perc_C', 'avg_hydrophobicity','avg_Flexibility','perc_H','perc_V','perc_W','perc_S', 'solvationFreeEnergy', 'perc_E']

# # choose features with importance by RF greater than 0.04
RF_features1 = ['num_dom', 'length', 'avg_hydrophobicity', 'avg_Flexibility', 'solvationFreeEnergy']
# # choose features with importance by RF greater than 0.04
# RF_features2 = ['num_dom', 'length','avg_hydrophobicity', 'avg_Flexibility', 'avg_solvAccess', 'avg_buriedProportion','avg_buriedRatio']
# # delete some redundant features based on correlation of avg_hydrophobicity and other features
# RF_features3 = ['num_dom', 'length','avg_hydrophobicity', 'avg_solvAccess']

# # delete some redundant features based on correlation of avg_Flexibility and other features
# RF_features4 = ['num_dom', 'length','avg_Flexibility', 'avg_solvAccess']
# # delete some redundant features based on correlation of avg_BuriedProportion and other features
# RF_features5 = ['num_dom', 'length','avg_solvAccess','avg_buriedProportion']

# # delete some redundant features based on correlation of avg_BuriedRatio and other features
# RF_features6 = ['num_dom', 'length','avg_solvAccess','avg_buriedRatio']

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
# # # 1.3 train the linear model on the training data
# # LR_clf_sample = linear_model.Lasso(alpha = 0.1)
# # LR_clf_sample.fit(X_sample_train,y_sample_train)
# # # Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,normalize=False, positive=False, precompute=False, random_state=None,selection='cyclic', tol=0.0001, warm_start=False)
# # # predict accuracy
# # LR_acc_sample= LR_clf_sample.score(X_sample_test,y_sample_test)
# # print("Accuracy: {:.4f}".format(LR_acc_sample))

# # -----------------------------------------------------------------------------#
# # 2.4 train the logistic regression model on the training data
# # LR_clf_sample = linear_model.LogisticRegression()
# # LR_clf_sample.fit(X_sample_train,y_sample_train)
# # Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,normalize=False, positive=False, precompute=False, random_state=None,selection='cyclic', tol=0.0001, warm_start=False)
# # predict accuracy
# # LR_acc_sample= LR_clf_sample.score(X_sample_test,y_sample_test)
# # print("Accuracy of logistic recression: {:.4f}".format(LR_acc_sample))

#-----------------------------------------------------------------------------#
# 2. model on Ab data
# 2.1 prepare the imputs of all the features for the regression model
y_Ab = Ab_data['num_dom']
# print(y_Ab.value_counts())
X_Ab = Ab_data.copy().drop(['num_dom'],axis=1)
print(X_Ab.shape)
print(X_Ab.columns)

# # draw correlation map
# corr_RF2 = X_Ab.corr()
# # generate a customed diverging colormap
# cmap = sns.diverging_palette(220,10,as_cmap = True)
# # draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr_RF2,vmax = 1, vmin = -1)
# plt.title('correlation heatmap of RFfeatureSet2')
# plt.ylabel('RFfeatureset2')
# plt.xlabel('RFfeatureset2')
# plt.show()
# print(corr_RF2)
# 2.2 randomly split the Ab dataset into training set and test set with 90%:10% ratio
X_Ab_train,X_Ab_test,y_Ab_train, y_Ab_test = cross_validation.train_test_split(X_Ab,y_Ab, test_size=0.1, random_state=0)

#-----------------------------------------------------------------------------#
# feature selection
# 1. Lasso
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_Ab,y_Ab)
model = SelectFromModel(lsvc, prefit=True)
X_Ab_new = pd.DataFrame(model.transform(X_Ab))
print(X_Ab_new.head())
print(X_Ab_new.shape)

# result: delete 4 features from the original set
# --------------------------------------------#
# 2. Removing features with low variance
sel = VarianceThreshold(threshold=0.6)
X_Ab_new = pd.DataFrame(sel.fit_transform(X_Ab))
print(X_Ab_new.head())
# result: delete 4 features with low variance
# --------------------------------------------#

# # randomly split the new Ab dataset into training set and test set with 90%:10% ratio
# # X_Ab_train_new,X_Ab_test_new,y_Ab_train_new, y_Ab_test_new = cross_validation.train_test_split(X_Ab_new,y_Ab, test_size=0.1, random_state=0)
#-----------------------------------------------------------------------------#
# # 2.3 train the linear model on the training data
# LR_clf_Ab = linear_model.Lasso(alpha = 0.1)
# LR_clf_Ab.fit(X_Ab_train,y_Ab_train)
# # Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,normalize=False, positive=False, precompute=False, random_state=None,selection='cyclic', tol=0.0001, warm_start=False)
# # predict accuracy
# LR_acc_Ab= LR_clf_Ab.score(X_Ab_test,y_Ab_test)
# print("Accuracy of linear regression: {:.4f}".format(LR_acc_Ab))

# # #-----------------------------------------------------------------------------#
# #2.4 train the logistic regression model on the training data
# Logistic_clf_Ab = linear_model.LogisticRegression()
# Logistic_clf_Ab.fit(X_Ab_train,y_Ab_train)

# # predict accuracy
# Logistic_acc_Ab= Logistic_clf_Ab.score(X_Ab_test,y_Ab_test)
# print("Accuracy of logistic regression: {:.4f}".format(Logistic_acc_Ab))

# #-----------------------------------------------------------------------------#
# # # # 2.42 train the logistic regression model on the training data(after feature selection)
# # # Logistic_clf_Ab_new = linear_model.LogisticRegression()
# # # Logistic_clf_Ab_new.fit(X_Ab_train_new,y_Ab_train_new)

# # # # predict accuracy
# # # Logistic_acc_Ab_new = Logistic_clf_Ab_new.score(X_Ab_test_new,y_Ab_test_new)
# # # print("Accuracy of logistic regression: {:.4f}".format(Logistic_acc_Ab_new))
# # #-----------------------------------------------------------------------------#
# # # 2.5 train the SGDclassifier on the training data
# # SGD_clf_Ab = linear_model.SGDClassifier()
# # SGD_clf_Ab.fit(X_Ab_train,y_Ab_train)

# # # predict accuracy
# # SGD_acc_Ab= SGD_clf_Ab.score(X_Ab_test,y_Ab_test)
# # print("Accuracy of SGD: {:.4f}".format(SGD_acc_Ab))
# # #-----------------------------------------------------------------------------#
# # 2.6 train the SVM model on the training data
# SVM_clf_Ab = svm.SVC(decision_function_shape='ovo')
# SVM_clf_Ab.fit(X_Ab_train,y_Ab_train)

# # predict accuracy
# SVM_acc_Ab= SVM_clf_Ab.score(X_Ab_test,y_Ab_test)
# print("Accuracy of SVM: {:.4f}".format(SVM_acc_Ab))

# # #--------------------------------#
# # linSVC_clf_Ab = svm.LinearSVC()
# # linSVC_clf_Ab.fit(X_Ab_train,y_Ab_train)

# # # predict accuracy
# # linSVC_acc_Ab= linSVC_clf_Ab.score(X_Ab_test,y_Ab_test)
# # print("Accuracy of linear SVC regression: {:.4f}".format(linSVC_acc_Ab))

# #-----------------------------------------------------------------------------#
# # 2.5 train the random forest classifier on the training data
# RF_clf_Ab = RandomForestClassifier(n_estimators=10)
# RF_clf_Ab.fit(X_Ab_train,y_Ab_train)

# # predict accuracy
# RF_acc_Ab= RF_clf_Ab.score(X_Ab_test,y_Ab_test)
# print("Accuracy of RF: {:.4f}".format(RF_acc_Ab))

# # draw feature importance
# plt.plot(RF_clf_Ab.feature_importances_)
# plt.title("Feature importance by Random Forest Model")
# plt.show()
# #-----------------------------------------------------------------------------#
# # 2.6 train the XGBclassifier on the training data
# xgb_clf_Ab = xgb.XGBClassifier()
# xgb_clf_Ab.fit(X_Ab_train,y_Ab_train)

# # predict accuracy
# xgb_acc_Ab= xgb_clf_Ab.score(X_Ab_test,y_Ab_test)
# print("Accuracy of xgb: {:.4f}".format(xgb_acc_Ab))

# # draw feature importance
# xgb.plot_importance(xgb_clf_Ab,title = 'Feature importance',xlabel = 'F score', ylabel = 'Features', grid = True )
# plt.show()
