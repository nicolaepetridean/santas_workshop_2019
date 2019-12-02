# linear algebra
import numpy as np

# data processing
import pandas as pd

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


def load_data():
    test_df = pd.read_csv("../data/test.csv")
    train_df = pd.read_csv("../data/train.csv")

    return (test_df, train_df)


def get_data_info(data_df):
    data_df.info()
    data_df.describe()


def analize_age(data_df):
    survived = 'survived'
    not_survived = 'not survived'
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    women = train_df[train_df['Sex'] == 'female']
    men = train_df[train_df['Sex'] == 'male']
    ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[0], kde=False)
    ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[0], kde=False)
    ax.legend()
    ax.set_title('Female')
    ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[1], kde=False)
    ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[1], kde=False)
    ax.legend()
    _ = ax.set_title('Male')

    plt.show()


if __name__ == "__main__":
    (test_df, train_df) = load_data()
    training_data_description = train_df.describe()
    print(training_data_description['Age'])
    print(training_data_description['Fare'])
    print(training_data_description['Parch'])
    print(training_data_description['Pclass'])
    print(training_data_description['Age'])
    print(training_data_description['SibSp'])

    analize_age(train_df)