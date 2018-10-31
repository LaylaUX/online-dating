 
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
m["body_image"] = m.body_type.map(body_image_mapping)

# Income-mapping: I've assigned numerical classes to income based on the following:
# The The median income for 2012 (the year of this dataset) was $28,213, so
# For $0 - 20,000, income class = 0 (below the meadian income)
# 20,0001 - 99,999, income class = 1
# 100,000 - 999,999, income class = 2
# 1,000,000+, income class = 3
# Again, these divisions were somewhat arbitrary

#%%
income_mapping = {20000: 0, 30000: 1, 40000: 1, 50000: 1, 60000: 1, 70000: 1,
                  80000: 1, 100000: 2, 150000: 2, 250000: 2, 500000: 2, 1000000: 3}

m["income_class"] = m.income.map(income_mapping)

m.columns.values
m.head()

#%%
f["income_class"] = f.income.map(income_mapping)
f.columns.values
f.head()

#%%
# Taking a look at preliminary graphs
plt.plot(m['income'], m['body_image'])
plt.xlabel("Income")
plt.ylabel("Body-image")
plt.title('Men')
plt.show()

#%%
plt.plot(f['income'], f['body_image'])
plt.xlabel("Income")
plt.ylabel("Body-image")
plt.title('Women')
plt.show()

# As expected, the men's graph is much more distributed as to income (x-axis)
# There's also more data for men, which makes for a busier graph

# Normalizing data
#%%
from sklearn.preprocessing import MinMaxScaler

m_data = m[['income', 'body_image', 'income_class']]
Mx = m_data.values
min_max_scaler = MinMaxScaler()
Mx_scaled = min_max_scaler.fit_transform(Mx)

#%%
f_data = f[['income', 'body_image', 'income_class']]
Fx = f_data.values
min_max_scaler = MinMaxScaler()
Fx_scaled = min_max_scaler.fit_transform(Fx)

#%%
m_data = pd.DataFrame(Mx_scaled, columns=m_data.columns)
f_data = pd.DataFrame(Fx_scaled, columns=f_data.columns)

m_data.head()
#%%
f_data.head()

# Ploting normalized data
#%%
plt.plot(f_data['income'], f_data['body_image'])
plt.xlabel("Income")
plt.ylabel("Body-image")
plt.title('Women - scaled')
plt.show()

#%%
plt.plot(m_data['income'], m_data['body_image'])
plt.xlabel("Income")
plt.ylabel("Body-image")
plt.title('Men - scaled')
plt.show()

# The normalized data graphs in the same way as the original data.

# Taking a look at the graphs using income class, instead
#%%
plt.plot(f_data['income_class'], f_data['body_image'])
plt.xlabel("Income Class")
plt.ylabel("Body-image")
plt.title('Women - scaled')
plt.show()

#%%
plt.plot(m_data['income_class'], m_data['body_image'])
plt.xlabel("Income Class")
plt.ylabel("Body-image")
plt.title('Men - scaled')
plt.show()

# As might be expected, the graphs now create a geometic pattern, with nodes
# I suspect, from looking at these graphs that it will not be possible to predict income based on body_image, or visa versa

 