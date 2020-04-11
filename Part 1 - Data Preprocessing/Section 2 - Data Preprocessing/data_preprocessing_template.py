# Data preprocessing

# Importing the librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
missing_values = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)
missing_values = missing_values.fit(X[:, 1:3])
X[:, 1:3] = missing_values.transform(X[:, 1:3])

# Encoding categorical data
""" Label encoder is not efficient for those kind of categorical data
as Country are not dissimilar, they cannot be rank as 1 will not be greater than 2 etc. """
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])

""" Those kind of categorical data are called dummy variable, and are encoded by splitting the column
into a table of dimension equal to the number of distinct values. """
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])],
                       remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
