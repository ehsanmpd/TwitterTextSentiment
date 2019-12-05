import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegressionCV

from sklearn.datasets import load_iris



# clf = LogisticRegressionCV(cv=5, random_state=0, max_iter=500).fit(X, y)
# clf.predict(X[:2, :])
# clf.predict_proba(X[:2, :]).shape
# print(clf.score(X, y))

# linear_model.LogisticRegressionCV(class_weight='balanced',
#                                                  scoring='roc_auc',
#                                                  n_jobs=FLAGS.n_jobs,
#                                                  max_iter=10000, verbose=1)

# lbfgs = Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm
# X.shape = (N,D), y.shape=(1,N)
def test_reg_logistic_regression(X_train,y_train,X_test,k_fold,max_iter):
    clf = LogisticRegressionCV(Cs=10, class_weight=None, cv=k_fold, dual=False,
                     fit_intercept=True, intercept_scaling=1.0, l1_ratios=None,
                     max_iter=max_iter, multi_class='auto', n_jobs=None,
                     penalty='l2', random_state=0, refit=True, scoring=None,
                     solver='lbfgs', tol=0.0001, verbose=0).fit(X_train, y_train)
    prediction = clf.predict(X_test)

    # print(clf.score(X, y))
    return prediction


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    print(y.shape, X.shape)

    # print(y)
    prediciton = test_reg_logistic_regression(X[21:150,:], y[21:150],X[1:20], k_fold=5,max_iter=500)
    print(y[1:20])
    print(prediciton)