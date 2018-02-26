#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',#'salary',
'expenses',
#'long_term_incentive',
'total_payments','bonus'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print("Number of individuals: "+str(len(data_dict.keys())))
print("Number of features: "+str(len(data_dict["METTS MARK"].keys())))
print("Number of total data points: "+str(len(data_dict["METTS MARK"].keys())*len(data_dict.keys())))
#print("POI: " +str(data_dict["METTS MARK"]["poi"]))
count_poi = 0
count_salary = 0
count_expenses = 0
count_long_term_incentive = 0
count_total_payments = 0
count_bonus = 0
for i in data_dict:
    count_poi += int(data_dict[i]["poi"])
    #print(data_dict[i]["salary"])
    if data_dict[i]["salary"] == "NaN":
        count_salary += 1
    if data_dict[i]["expenses"] == "NaN":
        count_expenses += 1
    if data_dict[i]["long_term_incentive"] == "NaN":
        count_long_term_incentive += 1
    if data_dict[i]["total_payments"] == "NaN":
        count_total_payments += 1
    if data_dict[i]["bonus"] == "NaN":
        count_bonus += 1
print("count_poi: " + str(count_poi))
print("count_salary: " + str(count_salary))
print("count_expenses: " + str(count_expenses))
print("count_long_term_incentive: " + str(count_long_term_incentive))
print("count_total_payments: " + str(count_total_payments))
print("count_bonus: " + str(count_bonus))

### Task 2: Remove outliers
del data_dict["TOTAL"]
### Task 3: Create new feature(s)
# for i in data_dict:
#     data_dict[i]["net_worth"] = 0
#     if data_dict[i]["salary"] != "NaN" and data_dict[i]["expenses"] != "NaN" and data_dict[i]["long_term_incentive"] != "NaN" and data_dict[i]["total_payments"] != "NaN":
#         data_dict[i]["net_worth"] += data_dict[i]["salary"] + data_dict[i]["long_term_incentive"] - data_dict[i]["expenses"] - data_dict[i]["total_payments"]
#     else:
#         data_dict[i]["net_worth"] = "NaN"
# for i in data_dict:
#     data_dict[i]["net_worth"] = 0
#     if data_dict[i]["expenses"] != "NaN" and data_dict[i]["bonus"] != "NaN" and data_dict[i]["total_payments"] != "NaN":
#         data_dict[i]["net_worth"] += data_dict[i]["bonus"] - data_dict[i]["expenses"] - data_dict[i]["total_payments"]
#     else:
#         data_dict[i]["net_worth"] = "NaN"
#features_list.append("net_worth")
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
### your code below
#print(features)
# import matplotlib.pyplot
# for i,point in enumerate(features):
#     expenses = point[1]
#     salary = point[0]
#     matplotlib.pyplot.scatter( salary, expenses )
# matplotlib.pyplot.xlabel("salary")
# matplotlib.pyplot.ylabel("expenses")
# matplotlib.pyplot.show()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
from time import time
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
clf = DecisionTreeClassifier(criterion = "entropy",min_samples_split=3,random_state=0, max_depth = 5)
t0 = time()
clf.fit(features_train,labels_train)
print("training time: " + str(round(time()-t0,3))+"s")
t1 = time()
pred = clf.predict(features_test)
print("training time: " + str(round(time()-t1,3))+"s")
print(accuracy_score(pred, labels_test))
print(clf.feature_importances_)

# clf = AdaBoostClassifier(learning_rate = 1)
# t0 = time()
# clf.fit(features_train,labels_train)
# print("training time: " + str(round(time()-t0,3))+"s")
# t1 = time()
# pred = clf.predict(features_test)
# print("training time: " + str(round(time()-t1,3))+"s")
# print(accuracy_score(pred, labels_test))
# print(clf.feature_importances_)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
