import numpy as np
import pandas as pd

df = pd.read_excel(r"../data/raw/Concrete_Data.xls")
print(df.shape)
y = df.iloc[:,-1]
x = df.iloc[:,:-1]
x.to_csv("../data/raw/x_raw.csv",index = False)
y.to_csv("../data/raw/y_raw.csv",index = False)
#normalize train by min-max normaliztion
for column in x.columns:
    x[column] = (x[column] - x[column].min())/(x[column].max() - x[column].min())
x.to_csv('../data/processed/x.csv',index = False)
y.to_csv('../data/processed/y.csv',index = False)