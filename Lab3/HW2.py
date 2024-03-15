#############################################################
#
# Name: 
# 
# Time Spent:
#
# ############################################################

import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
# import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


def scale(df: pd.DataFrame) -> pd.DataFrame:
    """ x' = (x - mean)/sd 
    Args:
        df (pd.DataFrame): Dataframe to scale 
    Returns:
        pd.DataFrame having standardized features
        
    Note: Only apply after steps 1 and 2"""
    
    nonLabel = list(filter(lambda x: x != 'RainTomorrow', df.columns))
    
    # We don't want to scale our prediction 
    subset = df[nonLabel]
    # Mapping feature to it's mean and sd
    means = dict(subset.mean())
    sds = dict(subset.std())

    # Loop through and do the math
    for col in means:
        df[col] = (df[col] - means[col])/sds[col]
        # df.head(10)
    return df





def preprocess(filename: str) -> pd.DataFrame: 
    """ Preprocess your data 

    Args:
        filename (str): Name of the csv file containing the data

    Returns: 
        pd.DataFrame: Dataframe with relevent preprocessing applied
        
    """
    df = pd.read_csv(filename)
    #handle na
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    string_cols = df.select_dtypes(include=['object']).columns
    df[string_cols] = df[string_cols].fillna("unknown")

    df[string_cols] = df[string_cols].apply(lambda x: pd.factorize(x)[0])
    # df = scale(df)
    # print(df.head(10))
    return df
    # pass

data = preprocess('Lab3_train.csv')





def fit_predict(train_fname: str, test_fname: str) -> np.array: 

    
    """ Fit a logistic regression model and return its predictions on test data 

    Args:
        train_fname (str): Name of the training file 
        test_fname (str): Name of the testing file
    Returns:
        np.array: Predictions of the model on test data

    Note: 
        Make sure you preprocess both your train and test data!"""


    train_data = preprocess(train_fname)
    test_data = preprocess(test_fname)
    X_train = train_data.drop(columns=['RainTomorrow','Evaporation'])  
    y_train = train_data['RainTomorrow']

    X_test = test_data.drop(columns=['RainTomorrow','Evaporation'])  

    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    #Predict the response for test dataset
    Y_pred = classifier.predict(X_test)
    # Y_pred = np.array()
    # print(Y_pred)
    return Y_pred


def score(test_fname: str, Y_pred: np.array) -> list[float]:
    test = preprocess(test_fname)
    Y = test[test.columns[test.columns.isin(['RainTomorrow'])]]

    precision = metrics.precision_score(Y, Y_pred)
    recall = metrics.recall_score(Y, Y_pred)
    f1 = metrics.f1_score(Y, Y_pred)

    return precision, recall, f1

Y_pred = fit_predict("Lab3_train.csv", "Lab3_valid.csv")
print(score('Lab3_valid.csv', Y_pred))

def main():
    """This function is for your own testing. It will not be called by the leaderboard."""
    pass


if __name__ == "__main__":
    main()
