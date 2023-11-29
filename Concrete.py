import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("concrete.csv")

X = df.drop(columns="strength")

y = df['strength']

lm = LinearRegression()
model = lm.fit(X,y)

saved_model = pickle.dumps(model)

with open('rf_cement.pkl', 'wb') as file:
    file.write(saved_model)