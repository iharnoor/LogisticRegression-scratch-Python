import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


# a) Create a function that implements (2.2).
def maxBeta(X, y, beta, n_iter):
    sum = 0.0
    for i in n_iter:
        part1 = exponentBetaCmp(X[i], beta) ** y[i]
        part2 = (1 - exponentBetaCmp(X[i], beta)) ** (1 - y[i])
        sum += math.log(part1 * part2)
    return sum


def exponentBetaCmp(X, beta):
    return 1 / (1 + np.exp(-np.dot(X, beta)))


#  part b and c are in Logistic Regression class.
# Parsing the Data
dataset = pd.read_csv('titanic_data.csv')
X_independent = dataset.iloc[:, 1:].values
y_dependent = dataset.iloc[:, 0].values

# d) Splitting the dataset into the Training set and Test set: Randomly 1:4 ratio
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_independent, y_dependent, test_size=0.20, random_state=0)

# Feature Scaling to bring the data to one scale
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set From
from Assignment2.LogisticRegression import LogisticRegression

classifier = LogisticRegression()
classifier.gradientAscent(X_train, y_train)

# Predicting the Test set results for the Test data after training with the train data
y_test_prediction = classifier.predict(X_test)

"""e) After running the training data with different values of eta, we found out that eta=0.1 or 0.2 seems to be the
best eta as it produces the best predictions with accuracy of 73.8%
Largest Beta= [-0.57436749 -0.97306453  1.36425123 -0.65789315 -0.36998717 -0.09503383]
Also, increasing the value of eta decreases the accuracy value. And decreasing the value of eta makes the complexity
  of the code low.
"""

# f) Making the Confusion Matrix to check the accuracy
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_test_prediction)
print('Confusion Matrix =', cm)
"""According to confusion matrix, accuracy = 100-((16+21)/(101+40))%
Accuracy =Eta =0.1 accuracy  = 73.8%
confusion matrix:
[[101  16]
[ 21  40]]
"""

# g) Prediciting my personal feature vector.
"""According to our Beta final vector, out of passenger class, family aboard and fare, our will mainly depend
on the Family aboard because that has the maximum beta coefficient. So changing family aboard will affect your
survival/prediction to more extent.
Also, changing Gender, Paren/Children Aboard or fare will decrease the chances of survival.
beta_coefficient =[-0.57436749 -0.97306453  1.36425123 -0.65789315 -0.36998717 -0.09503383]
"""
personal_test_data = [[1, 1, 20, 0, 0, 10]]
personal_test_data = sc.transform(personal_test_data)

my_test_prediction = classifier.predict(personal_test_data)
print('Predictions of my Personal data vector', end='')
if (my_test_prediction == 0):
    print("You will sink")
else:
    print("You will survive")

# h) 3 features that most affect survival
"""According to the beta_coefficient matrix our prediction will mainly depend on Sex, Fare and Parents/Children Aboard
because the beta value associated with these values are the largest. So survival will affected to great extent
on changing the values of these 3 variables.
"""

plt.scatter(X_independent[:, 2], y_dependent, color='red')
plt.xlabel('Age')
plt.ylabel('Survived or Not')
plt.show()

plt.scatter(X_independent[:, 4], y_dependent, color='blue')
plt.xlabel('Family/Children Aboard')
plt.ylabel('Survived or Not')
plt.show()

plt.scatter(X_independent[:, 5], y_dependent, color='green')
plt.xlabel('Fare')
plt.ylabel('Survived or Not')
plt.show()
