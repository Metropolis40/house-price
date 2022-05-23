# %%
import os
os.getcwd()
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression



# %%

df = pd.read_csv (r'.\train.csv')


# %%
df.head(10)

# %%
data1 =  pd.get_dummies(df)
print(type(data1))

# %%
data1.shape

# %%
data1.head(10)

# %%
data1_estimate = data1.iloc[1:1200,]

# %%
data1_estimate.shape

# %%
model = LinearRegression( n_jobs =-1)

# %%
x = data1_estimate.loc[:,("LotArea", "YearRemodAdd")]
y =  data1_estimate.loc[:,"SalePrice"]

model.fit(x, y )
                 

# %%
r_sq = model.score(x, y )
print(f"coefficient of determination: {r_sq}")
 

# %%
print(f"intercept: {model.intercept_}")
intercept: 5.633333333333329

print(f"slope: {model.coef_}")

# %%
print(model.coef_)
print(model.intercept_)
print(x)

# %%
y_pred = model.intercept_ + model.coef_ * x
print(y_pred)

# %%



