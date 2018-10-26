 
#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Creating a dataframe and exploring the data
df = pd.read_csv("profiles.csv")
df.columns.values

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

# Q: Is there a relationship between body-image and income?

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

#Creating two new datarames, one each for m and f
#%%
f = body_and_income.loc[df['sex'] == 'f']
f.columns.values
f.count

#%%
m = body_and_income.loc[df['sex'] == 'm']
m.columns.values
m.count

#cleaning the data
#%%
m = m.dropna()
m.count
m.columns.values

#%%
m_undisclosed_income = m.loc[m['income'] == -1]
m_undisclosed_income.count

#%%
m = m.loc[m['income'] != -1]
m.count

#%%
f = f.dropna()
f.count
f.columns.values

#%%
f_undisclosed_income = f.loc[f['income'] == -1]
f_undisclosed_income.count

#%%
f = f.loc[f['income'] != -1]
f.count

# Taking a look at the body-type and income splits in the m and f dataframes
#%%
m.body_type.value_counts()

#%%
f.body_type.value_counts()

#%%
m.income.value_counts()

#%%
f.income.value_counts()

#%%
m = m.drop(columns=['sex'])
f = f.drop(columns=['sex'])

# Augmenting data - transforming body-type to body-image
# A note on methodology: 
# I assgined labels according to the relative positive/negative connotation of body-type labels
# This was an unscientific process based on my expeirence as a fiction author
# 2 = curvy, fit, thin, athletic, full-figured, jacked
# 1 = average, a little extra, skinny
# 0 = used up, overweight, rather not say

#%%

body_image_mapping = {"average": 1, "curvy": 2, "fit": 2, "thin": 2, "athletic": 2, "full figured": 2, "a little extra": 1, "skinny": 1, "jacked": 2, "used up": 0, "overweight": 0, "rather not say": 0}

f["body_image"] = f.body_type.map(body_image_mapping)

#%%
f.columns.values
f.head()

#%%
m["body_image"] = m.body_type.map(body_image_mapping)
m.columns.values
m.head()

# Normalizing data
