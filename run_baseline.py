import numpy as np
import pandas as pd
import pickle

from implementation import *

# from sklearn import linear_model
from sklearn.linear_model import LogisticRegressionCV

from sklearn.datasets import load_iris

import nltk
import ssl
from nltk import word_tokenize,sent_tokenize


# clf = LogisticRegressionCV(cv=5, random_state=0, max_iter=500).fit(X, y)
# clf.predict(X[:2, :])
# clf.predict_proba(X[:2, :]).shape
# print(clf.score(X, y))

# linear_model.LogisticRegressionCV(class_weight='balanced',
#                                                  scoring='roc_auc',
#                                                  n_jobs=FLAGS.n_jobs,
#                                                  max_iter=10000, verbose=1)

# X.shape = (N,D), y.shape=(1,N)
def test_reg_logistic_regression(X_train,y_train,X_test,k_fold,max_iter):

    """
    It runs regularized logistic regression with cross validation of k_fold and maximum iteration max_iter.
    The solver used here is Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm (L-BFGS).
    """

    clf = LogisticRegressionCV(Cs=10, class_weight=None, cv=k_fold, dual=False,
                     fit_intercept=True, intercept_scaling=1.0, l1_ratios=None,
                     max_iter=max_iter, multi_class='auto', n_jobs=None,
                     penalty='l2', random_state=0, refit=True, scoring=None,
                     solver='lbfgs', tol=0.0001, verbose=0).fit(X_train, y_train)
    prediction = clf.predict(X_test)

    # print(clf.score(X, y))
    return prediction


def step1(PIK_train, PIK_test):
    train_neg = pd.read_csv("data/twitter-datasets/train_neg.txt", sep="asdfgsdgsdfgsgsdg", header=None,
                            engine='python')
    train_pos = pd.read_csv("data/twitter-datasets/train_pos.txt", sep="asdfgsdgsdfgsgsdg", header=None,
                            engine='python')

    test_data = pd.read_csv("data/twitter-datasets/test_data.txt", sep="asdfgsdgsdfgsgsdg", header=None,
                            engine='python')

    # print(train_neg.loc[10])
    train_neg[0] = preprocessing(train_neg[0])
    train_pos[0] = preprocessing(train_pos[0])

    test_data[0] = preprocessing(test_data[0])
    #
    train_neg[0] = train_neg[0].apply(nltk.word_tokenize)
    train_pos[0] = train_pos[0].apply(nltk.word_tokenize)

    test_data[0] = test_data[0].apply(nltk.word_tokenize)
    # print(train_neg[0])
    #
    df_list_train = [train_neg[0], train_pos[0]]
    full_df_train = pd.concat(df_list_train, ignore_index=True)
    #
    # shuffling the rows of dataframe
    full_df_train = full_df_train.sample(frac=1)
    full_df_test = test_data.sample(frac=1)
    # print(full_df_test.shape,full_df_test)
    with open(PIK_train, "wb") as f:
        pickle.dump(full_df_train, f)
    with open(PIK_test, "wb") as f:
        pickle.dump(full_df_test, f)

if __name__ == '__main__':
    print("Loading Data...")

    # try:
    #     _create_unverified_https_context = ssl._create_unverified_context
    # except AttributeError:
    #     pass
    # else:
    #     ssl._create_default_https_context = _create_unverified_https_context
    # nltk.download('punkt')

    PIK_train = "step1_train.dat"
    PIK_test = "step1_train.dat"
    # step1(PIK_train,PIK_test)

    with open(PIK_train, "rb") as f:
        full_df_train = pickle.load(f)
    with open(PIK_test, "rb") as f:
        full_df_test = pickle.load(f)

    print(full_df_train)

    # train_pos, train_neg, test, glove_embedings = load_data()
    # print(train_pos)
    # print("Data Loaded...")
    exit(0)
    X, y = load_iris(return_X_y=True)
    print(y.shape, X.shape)

    # print(y)
    prediciton = test_reg_logistic_regression(X[21:150,:], y[21:150],X[1:20], k_fold=5,max_iter=500)
    print(y[1:20])
    print(prediciton)