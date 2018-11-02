 
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

#%%
# Creating a new column with total essay length
essay_cols = ["essay0", "essay1", "essay2", "essay3",
              "essay4", "essay5", "essay6", "essay7", "essay8", "essay9"]

all_essays = df[essay_cols].replace(np.nan, '', regex=True)
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)


df["essay_len"] = all_essays.apply(lambda x: len(x))

# Creating a new dataframe with the data I want: sex, body-type, and income
#%%
body_and_income = df[['sex', 'body_type', 'income', 'essay_len']].copy()

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

'''A note on methodology: 
I assgined labels according to the relative positive/negative connotation of body-type labels
This was an unscientific process based on my expeirence as a fiction author
2 = curvy, fit, thin, athletic, full-figured, jacked
1 = average, a little extra, skinny
0 = used up, overweight, rather not say'''

#%%

body_image_mapping = {"average": 1, "curvy": 2, "fit": 2, "thin": 2, "athletic": 2, "full figured": 2, "a little extra": 1, "skinny": 1, "jacked": 2, "used up": 0, "overweight": 0, "rather not say": 0}
f["body_image"] = f.body_type.map(body_image_mapping)

#%%
m["body_image"] = m.body_type.map(body_image_mapping)

'''Income-mapping: I've assigned numerical classes to income based on the following:
The The median income for 2012 (the year of this dataset) was $28,213, so
For $0 - 20,000, income class = 0 (below the meadian income)
20,0001 - 99,999, income class = 1
100,000 - 999,999, income class = 2
1,000,000+, income class = 3
Again, these divisions were somewhat arbitrary.'''

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
# Taking a look at some preliminary graphs
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

m_data = m[['income', 'essay_len', 'body_image', 'income_class']]
Mx = m_data.values
min_max_scaler = MinMaxScaler()
Mx_scaled = min_max_scaler.fit_transform(Mx)

#%%
f_data = f[['income', 'essay_len', 'body_image', 'income_class']]
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

 # Performing single variable linear regression:
#%%

mX = m_data['body_image']
my = m_data['income']

mX = mX.values.reshape(-1, 1)
print(mX)

#%%
fX = f_data['body_image']
fy = f_data['income']
fX = fX.values.reshape(-1, 1)
print(fX)

#%%
from sklearn import linear_model
from sklearn.model_selection import train_test_split

mX_train, mX_test, my_train, my_test = train_test_split(mX, my, test_size=0.2, random_state=1)

mX_train, mX_val, my_train, my_val = train_test_split(mX_train, my_train, test_size=0.2, random_state=1)

regr = linear_model.LinearRegression()

model = regr.fit(mX_train, my_train)

my_predict = regr.predict(mX_test)


print("Train score:")
print(regr.score(mX_train, my_train))
print("Test score:")
print(regr.score(mX_test, my_test))

residuals = my_predict - my_test

plt.scatter(my_predict, residuals, alpha=0.4)
plt.title('Residual Analysis - Men')

plt.show()

#%%
fX_train, fX_test, fy_train, fy_test = train_test_split(
    fX, fy, test_size=0.2, random_state=1)

fX_train, fX_val, fy_train, fy_val = train_test_split(
    fX_train, fy_train, test_size=0.2, random_state=1)

regr = linear_model.LinearRegression()

model = regr.fit(fX_train, fy_train)

fy_predict = regr.predict(fX_test)


print("Train score:")
print(regr.score(fX_train, fy_train))
print("Test score:")
print(regr.score(fX_test, fy_test))

residuals = fy_predict - fy_test

plt.scatter(fy_predict, residuals, alpha=0.4)
plt.title('Residual Analysis - Women')

plt.show()

'''The train scores and test scores are wildly different.
I can't even see a point in validating the model.
I will, however, test it with income predicting body image, instead.'''

'''#%%

my2 = m_data['body_image']
mX2 = m_data['income']

mX2 = mX2.values.reshape(-1, 1)
print(mX)


#%%
mX_train, mX_test, my_train, my_test = train_test_split(
    mX2, my2, test_size=0.2, random_state=1)

mX_train, mX_val, my_train, my_val = train_test_split(
    mX_train, my_train, test_size=0.2, random_state=1)

regr = linear_model.LinearRegression()

model = regr.fit(mX_train, my_train)

my_predict = regr.predict(mX_test)


print("Train score:")
print(regr.score(mX_train, my_train))
print("Test score:")
print(regr.score(mX_test, my_test))

residuals = my_predict - my_test

plt.scatter(my_predict, residuals, alpha=0.4)
plt.title('Residual Analysis - Men (income predicting body image)')

plt.show()'''

#%%
# As I expected, this doesn't appear any more accurate.

'''At this point, I went back up in my code and added a column for total essay length.
Since I already see no correlation between income and body_image, 
it would make sense to perform single variable linear regression again,
to check for a correlation between essay-length and body-image or income.
However, the assignmnent calls for two regression approaches, 
so I will now perform multiple linear regression.'''

# To save time, I will do this fior men only (the larger data sample).
# Also, to avoid confusion, I'm commenting out my last experiment.

#%%

X_multilinear = m_data[['body_image', 'essay_len']]
y_multilinear = m_data['income']

#%%

X_train, X_test, y_train, y_test = train_test_split(
    X_multilinear, y_multilinear, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=1)

regr = linear_model.LinearRegression()

model = regr.fit(X_train, y_train)

y_predict = regr.predict(X_test)


print("Train score:")
print(regr.score(X_train, y_train))
print("Test score:")
print(regr.score(X_test, y_test))

residuals = y_predict - y_test

plt.scatter(y_predict, residuals, alpha=0.4)
plt.title('Residual Analysis - Men (multi-variable)')

plt.show()

# Again, the train and test sets show wildly different scores.
# There doesn't seem to be a correlation and validating the model would not be productive use of time.

# K-means for regression
#%%
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
%matplotlib inline

rmse_val = []  # to store rmse values for different k
for K in range(200):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors=K)

    model.fit(X_train, y_train)  # fit the model
    pred = model.predict(X_test)  # make prediction on test set
    error = sqrt(mean_squared_error(y_test, pred))  # calculate rmse
    rmse_val.append(error)  # store rmse values
    print('RMSE value for k= ', K, 'is:', error)

#%%
#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val)  # elbow curve
curve.plot()

#%%
K = 15
model = neighbors.KNeighborsRegressor(n_neighbors=K)

model.fit(X_train, y_train)  # fit the model
pred = model.predict(X_test)  # make prediction on test set
error = sqrt(mean_squared_error(y_test, pred))  # calculate rmse
print('Error for k= ', K, 'is:', error)

# Model accuracy
#%%
val = model.predict(X_val)
error = sqrt(mean_squared_error(y_val, val))  # calculate rmse
print('Error for k= ', K, 'is:', error)

# These error values are quite similar, but also quite high.

#%%
# Trying classification approaches

# KNN Classification
# Setting up the data

fX_classification = f_data['body_image']
fy_classification = f_data['income_class']
fX_classification = fX_classification.values.reshape(-1, 1)
print(fX)

#%%
mX_classification = m_data['body_image']
my_classification = m_data['income']
mX_classification = mX_classification.values.reshape(-1, 1)
print(mX)

#%%
mX_train, mX_test, my_train, my_test = train_test_split(
    mX_classification, my_classification, test_size=0.2, random_state=1)

mX_train, mX_val, my_train, my_val = train_test_split(
    mX_train, my_train, test_size=0.2, random_state=1)

print(len(mX_train))
print(len(mX_test))
print(len(mX_val))

#%%
fX_train, fX_test, fy_train, fy_test = train_test_split(
    fX_classification, fy_classification, test_size=0.2, random_state=1)

fX_train, fX_val, fy_train, fy_val = train_test_split(
    fX_train, fy_train, test_size=0.2, random_state=1)

print(len(fX_train))
print(len(fX_test))
print(len(fX_val))

# There's just not a whole lot of female data. I'll move forward with the male data.

#%%
from sklearn.neighbors import KNeighborsClassifier

k = 1
accuracies = []

for k in range(1, 101):
  classifier = KNeighborsClassifier(n_neighbors=k)
  classifier.fit(mX_train, my_train)
  accuracies.append(classifier.score(mX_val, my_val))
  k += 1


#%%
import matplotlib.pyplot as plt

k_list = range(1, 101)

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Body Image as Predictor of Income (KNN validation accuracy)")
plt.show()









