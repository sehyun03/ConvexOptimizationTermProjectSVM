import numpy as np
import cPickle
import gzip
import time
import timeit

from sklearn import svm
from sklearn import metrics

# Read MNIST file

def read_mnist(mnist_file):
    """
    Reads MNIST data.
    """
    f = gzip.open(mnist_file, 'rb')
    train_data, val_data, test_data = cPickle.load(f)
    f.close()
    train_X, train_Y = train_data
    val_X, val_Y = val_data
    test_X, test_Y = test_data    
    return train_X, train_Y, val_X, val_Y, test_X, test_Y

# Get training, validation, testing data
train_X, train_Y, val_X, val_Y, test_X, test_Y = read_mnist('mnist.pkl.gz')

# C and gamma give best perform: C = 10 & gamma = 0.01 with Eval = 0.0165

# Use the merge dataset of Dtrain and Dval to have more data
final_train_X = train_X
final_train_X = np.vstack([final_train_X,val_X])
final_train_Y = train_Y
final_train_Y = np.hstack([final_train_Y,val_Y])

# Use SVM to get the best model
final_start_time_RBF_train = time.time()
final_RBF_clf = svm.SVC(C= 10, kernel='rbf', gamma =0.01, verbose=True)
final_RBF_clf.fit(train_X,train_Y)
final_end_time_RBF_train = time.time() - final_start_time_RBF_train

# Print Time
print 'training Dtrain for final H complete at %s seconds' %(final_end_time_RBF_train)

# Test the final model get from C and gamma below by using Dtest:

# Predict in Dfinal_train
# final_RBF_Y_comma_train = final_RBF_clf.predict(final_train_X)

# Predict in Dtest
final_RBF_Y_comma_test = final_RBF_clf.predict(test_X)

# Compute Ein of Dfinal_train and Etest of Dtest

# Compute Ein of Dfinal_train
# RBF_Ein_final_train = 1 - metrics.accuracy_score(final_train_Y, final_RBF_Y_comma_train)

# Compute Etest of Dtest
RBF_Ein_final_test = 1 - metrics.accuracy_score(test_Y, final_RBF_Y_comma_test)

# Print Result
#print 'Ein = ',Ein_final_train
print 'Etest = ', RBF_Ein_final_test
