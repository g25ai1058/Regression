import numpy as np
import statsmodels.api as sm

def create_data():
    X = np.array([1,2,3,4,5,6,7,8])
    y = np.array([2,4,5,4,5,6,5,6])
    return X, y

def add_constant(X):
    return sm.add_constant(X)

def train_model(X, y):
    model = sm.OLS(y, X).fit()
    return model
