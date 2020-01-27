import numpy as np
import pandas as pd
import pickle
import csv


#from implementation import *
#from helpers import *

from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.insert(1,"data/")
sys.path.insert(1,"data_preprocessing/")
from read_and_preprocess import preprocessing

import nltk
import ssl


def test_reg_logistic_regression(X_train,y_train,X_test_local, X_test,k_fold,max_iter):

    """
    It runs regularized logistic regression with cross validation of k_fold and maximum iteration max_iter.
    """
    print('------------------------------------------------------')
    print("Testing Regularized Logistic Regression with %d -fold cross-validation..." %(k_fold))
    # running regularized logistic regression with k_fold cross validation
    clf = LogisticRegressionCV(Cs=10, class_weight=None, cv=k_fold, dual=False,
                     fit_intercept=True, intercept_scaling=1.0, l1_ratios=None,
                     max_iter=max_iter, multi_class='auto', n_jobs=None,
                     penalty='l2', random_state=0, refit=True, scoring=None,
                     solver='liblinear', tol=0.0001, verbose=0).fit(X_train, y_train)

    # finding the accuracy of the model based on the partial training data
    prediction = clf.predict(X_test_local)
    acc = 100 * (np.sum(prediction == y_test)) / len(y_test)
    print('Accuracy of the model: {0:f}'.format(acc))
    print('------------------------------------------------------')
    # creating the prediction of the test data
    prediction = clf.predict(X_test)
    return prediction


def test_SVM(X_train,y_train,X_test_local, X_test):

    """
    It runs SVM.
    """

    print('------------------------------------------------------')
    print("Testing SVM...")
    clf = LinearSVC(random_state=0, tol=1e-5)
    # fitting the train data into the SVM model
    clf.fit(X_train, y_train)
    # finding the accuracy of the model based on the partial training data
    prediction = clf.predict(X_test_local)
    acc = 100 * (np.sum(prediction == y_test)) / len(y_test)
    print('Accuracy of the model: {0:f}'.format(acc))
    print('------------------------------------------------------')
    # creating the prediction of the test data
    prediction = clf.predict(X_test)
    return prediction


def generate_baseline_train_test_data(NEG_TRAIN_PATH,POS_TRAIN_PATH,TEST_PATH,TO_SAVE_TRAIN, TO_SAVE_TEST):

    # try:
    #     _create_unverified_https_context = ssl._create_unverified_context
    # except AttributeError:
    #     pass
    # else:
    #     ssl._create_default_https_context = _create_unverified_https_context
    # nltk.download('punkt')

    print('Generating data for baseline models...')

    # reading negative and positive input twitter train data
    train_neg = pd.read_csv(NEG_TRAIN_PATH, sep="asdfgsdgsdfgsgsdg", header=None,
                            engine='python')
    train_pos = pd.read_csv(POS_TRAIN_PATH, sep="asdfgsdgsdfgsgsdg", header=None,
                            engine='python')
    # reading twitter test data
    test_data = pd.read_csv(TEST_PATH, sep="asdfgsdgsdfgsgsdg", header=None,
                            engine='python')

    # preprocessing of the train and test data
    train_neg = preprocessing(train_neg)
    train_pos = preprocessing(train_pos)

    test_data = preprocessing(test_data)
    #tokenizing the train and test data; changing the strings into word list
    train_neg[0] = train_neg[0].apply(nltk.word_tokenize)
    train_pos[0] = train_pos[0].apply(nltk.word_tokenize)

    test_data[0] = test_data[0].apply(nltk.word_tokenize)

    #merging positive and negative tweets and converting to list
    df_list_train = [train_neg[0], train_pos[0]]
    full_df_train = pd.concat(df_list_train, ignore_index=True)

    df_list_test = [test_data[0]]
    full_df_test = pd.concat(df_list_test, ignore_index=True)

    #creating and merging training y matrix for negative and positive tweets: 0 for negative and 1 for positive
    y_neg = np.zeros(len(train_neg))
    y_pos = np.ones(len(train_pos))
    y_t = np.concatenate((y_neg, y_pos), axis=0)

    # creating features of each tweet based on its frequency in the train data
    full_df_train = full_df_train.map(lambda x: ' '.join(x))
    vectorizer = TfidfVectorizer(stop_words='english', min_df=5)
    train_data = vectorizer.fit_transform(list(full_df_train))

    # creating train and test matrixes for model training and local testing
    X_train, X_test, y_train, y_test = train_test_split(train_data, y_t, test_size=0.2, random_state=42)

    # creating features of each tweet based on its frequency in the test data
    full_df_test = full_df_test.map(lambda x: ' '.join(x))
    test_data_final = vectorizer.transform(list(full_df_test))

    return X_train, X_test, y_train, y_test, test_data_final
    
def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


if __name__ == '__main__':
    NEG_TRAIN_PATH = "data/twitter-datasets/train_neg_full.txt"
    POS_TRAIN_PATH = "data/twitter-datasets/train_pos_full.txt"
    TEST_PATH = "data/twitter-datasets/test_data.txt"

    TO_SAVE_TRAIN = 'train.npz'
    TO_SAVE_TEST = 'test.arr'

    X_train, X_test, y_train, y_test, test_data_final = generate_baseline_train_test_data(NEG_TRAIN_PATH,
                                                                                                  POS_TRAIN_PATH,
                                                                                                  TEST_PATH,
                                                                                                  TO_SAVE_TRAIN,
                                                                                                  TO_SAVE_TEST)

    print("Data generated...")

    # testing regularized logistic regression
    prediciton = test_reg_logistic_regression(X_train=X_train, y_train=y_train, X_test_local=X_test, X_test=test_data_final, k_fold=5, max_iter=500)
    # write the prediction output of regularized logistic regression
    prediciton[prediciton == 0] = -1
    ids = range(1, 1 + len(prediciton))
    create_csv_submission(ids, prediciton, "baseline_models/prediction_RLR.csv")

    # testing SVM
    prediciton = test_SVM(X_train=X_train, y_train=y_train, X_test_local=X_test, X_test=test_data_final)
    # write the prediction output of SVM
    prediciton[prediciton == 0] = -1
    ids = range(1, 1 + len(prediciton))
    create_csv_submission(ids, prediciton, "baseline_models/prediction_SVM.csv")
