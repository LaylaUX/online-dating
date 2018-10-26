 
#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Creating a dataframe and exploring the data
df = pd.read_csv("profiles.csv")
print(df.columns.values)

#%%
df.essay0.head()
df.essay7.head()
df.income.head()
df.last_online.head()
df.body_type.head()

#%%
#  Exploring age distribution
plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()

# Q: Is there a relationship between (self-perceptioon of) body-type and income?

# Taking a look at the body type and income type distributions...
#%%
df.sex.value_counts()

#%%
df.body_type.value_counts()

#%%
df.income.value_counts()

# Creating a new dataframe with the data I want: sex, body-type, and income
#%%
body_and_income = df[['sex', 'body_type', 'income']].copy()

# Checking to make sure that worked (it did)
print(body_and_income.columns.values)

#%%
f = body_and_income.loc[df['sex'] == 'f']

# Checking to see if it worked (Yup, looks good)
f.columns.values
f.count
body_and_income.count

#%%
m = body_and_income.loc[df['sex'] == 'm']
m.columns.values
m.count
