import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv(".\\nba_logreg.csv")

Y = df.TARGET_5Yrs
Y = Y.to_numpy()

Features = df.columns[1:20]

X = df[Features]

X_values = X.values

for x in np.argwhere(np.isnan(X_values)):
    X_values[x]=0.0
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_values)   

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, test_X, train_y, test_y = train_test_split(X_scaled, Y, test_size=0.1)
# Define model
model1 = LogisticRegression()
#model2 = LinearSVC(C=2.0)
#model3 = RandomForestClassifier(criterion="entropy",n_estimators=300,random_state=42)
# Fit model(model trained and ready to make predictions)
model1.fit(train_X, train_y)
#model2.fit(train_X, train_y)
#model3.fit(train_X, train_y)


def score_classifier(dataset,classifier,labels):

    """
    performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
    :param dataset: the dataset to work on
    :param classifier: the classifier to use
    :param labels: the labels used for training and validation
    :return:
    """

    kf = KFold(n_splits=3,random_state=50,shuffle=True)
    confusion_mat = np.zeros((2,2))
    recall = 0
    for training_ids,test_ids in kf.split(dataset):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set,training_labels)
        predicted_labels = classifier.predict(test_set)
        confusion_mat+=confusion_matrix(test_labels,predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
    recall/=3
    print(confusion_mat)
    print(recall)


score_classifier(X_scaled,model1,Y)

#model.predict_proba(test_X)


# save the model to disk
pickle.dump(model1, open("ml_model.sav", "wb"))
pickle.dump(scaler, open("scaler.sav", "wb"))


# load the model from disk
#loaded_model = joblib.load(filename)
#result = loaded_model.score(test_X, test_y)
#print(result)