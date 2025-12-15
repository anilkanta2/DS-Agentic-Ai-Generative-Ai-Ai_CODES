import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"D:\Data Science Classes\Daily Classes\DS (ML 12-)\12Dec-Dataset\Data.csv")

x=dataset.iloc[:,:-1].values
#depandent varibale
y=dataset.iloc[:,3].values

#Data cleaning
from sklearn.impute import SimpleImputer
imputer=SimpleImputer()

imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])


#categorical to number
from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)
