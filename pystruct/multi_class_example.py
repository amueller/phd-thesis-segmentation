from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC

from pystruct.models import CrammerSingerSVMModel
from pystruct.learners import OneSlackSSVM

# Load three class iris data.
iris = load_iris()
X, y = iris.data, iris.target

# PyStruct interface
model = CrammerSingerSVMModel()
one_slack_svm = OneSlackSSVM(model)
one_slack_svm.fit(X, y)

# scikit-learn interface for liblinear
libsvm = LinearSVC(multi_class='crammer_singer')
libsvm.fit(X, y)
