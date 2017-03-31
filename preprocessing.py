import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import ensemble
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from collections import Counter

plt.style.use('ggplot')
LISTINGSFILE = 'data/listings.csv'

cols = ['price',
        'accommodates',
        'bedrooms',
        'beds',
        'neighbourhood_cleansed',
        'room_type',
        'cancellation_policy',
        'instant_bookable',
        'reviews_per_month',
        'number_of_reviews',
        'availability_30',
        'review_scores_rating'
        ]

# read the file into a dataframe
df = pd.read_csv(LISTINGSFILE, usecols=cols)
print df.head()

# compute the distribution of house neighbourhood
nb_counts = Counter(df.neighbourhood_cleansed)
tdf = pd.DataFrame.from_dict(nb_counts, orient='index').sort_values(by=0)
#tdf.plot(kind='bar')

#plt.show()

# how many missing values for each feature
naNumber = df[df.number_of_reviews == 0].isnull().sum(0)
print naNumber

'''
The reviews_per month and the review_scores_rating have most missing value data, which caused by no review on them
'''

# deal with missing value

# first fixup reviews_per_month where there are no reviews
# fill the review_scores_rating with average
df['reviews_per_month'].fillna(0, inplace=True)
#print df['review_scores_rating'].mean()

# show the distribution of review score by hist
df['review_scores_rating'].hist()
#plt.show()
df['review_scores_rating'].fillna(df['review_scores_rating'].mean(), inplace=True)

# just drop rows with bad/weird values
# (we could do more here)
print df.shape
df = df[df.bedrooms != 0]
df = df[df.beds != 0]
df = df[df.price != 0]
df = df.dropna(axis=0)

print df.shape
#print sum(df.bedrooms == 1)

# only analyse one type data (or clustering the similar type and analyze them one by one)
df = df[df.bedrooms == 1]

# remove the $ from the price and convert to float
df['price'] = df['price'].replace('[\$,)]', '', regex=True).replace('[(]', '-', regex=True).astype(np.float64)

print df.info()
print df.head()
'''
neighbourhood_cleased, root_type, bedrooms, beds, accommodates, instant_bookable, cancellation_policy can be category 
value, then I have to encoding some of them
'''
def howManyType(column):
        return len(set(column))

## check the categories of each features

print df.apply(set, axis=0)
print df.apply(howManyType, axis=0)
# get feature encoding for categorical variables
n_dummies = pd.get_dummies(df.neighbourhood_cleansed)
rt_dummies = pd.get_dummies(df.room_type)
xcl_dummies = pd.get_dummies(df.cancellation_policy)

# convert boolean column to a single boolean value indicating whether this listing has instant booking available
ib_dummies = pd.get_dummies(df.instant_bookable, prefix="instant")
ib_dummies = ib_dummies.drop('instant_f', axis=1)


# replace the old columns with our new one-hot encoded ones
alldata = pd.concat((df.drop(['neighbourhood_cleansed', 'room_type', 'cancellation_policy','instant_bookable'], axis=1),
                     n_dummies.astype(int), rt_dummies.astype(int), xcl_dummies.astype(int),
                     ib_dummies.astype(int)), axis=1)

allcols = alldata.columns

scattercols = ['price',
               'accommodates',
               'number_of_reviews',
               'reviews_per_month',
               'beds',
               'availability_30',
               'review_scores_rating']
# axs = pd.scatter_matrix(alldata[scattercols],
#                        figsize=(12, 12), c='red')

# axs = sns.pairplot(df, vars=scattercols, hue='room_type', kind='reg')
# plt.show()

# plot the scatter plot and correlation to find the correlation between features
#ax1 = sns.pairplot(alldata, vars=scattercols, kind='reg')

# fig, ax = plt.subplots(figsize=(9, 9))
# sns.heatmap(alldata[scattercols].corr(), annot=True, linewidths=.5, ax=ax)
# plt.show()

# fig1, ax1 = plt.subplots(figsize=(8, 8))
# sns.barplot(x='instant_bookable', y='price', data=df)
# plt.show()


# test the learning model
rs = 1
        #SVR(kernel='rbf', C=1e3, gamma=0.1), SVR(kernel='linear', C=1e3), SVR(kernel='poly', C=1e3, degree=3)]
ests_labels = np.array(['Linear', 'Ridge', 'Lasso', 'ElasticNet', 'BayesRidge', 'OMP', 'svr_rbf', 'svr_lin'])#, 'svr_poly'])
errvals = np.array([])

ests = {
    'Linear': linear_model.LinearRegression(),
    'Ridge': linear_model.Ridge(),
    'Lasso': linear_model.Lasso(),
    'ElasticNet': linear_model.ElasticNet(),
    'BayesRidge': linear_model.BayesianRidge(),
    'OMP': linear_model.OrthogonalMatchingPursuit(),
    'SVR_rbf': SVR(kernel='rbf'),
    'SVR_lin': SVR(kernel='linear')
    #,'SVR_poly': SVR(kernel='poly')
}
X_train, X_test, y_train, y_test = train_test_split(alldata.drop(['price'], axis=1),
                                                    alldata.price, test_size=0.2, random_state=20)

for name, e in ests.items():
    e.fit(X_train, y_train)
    this_err = metrics.median_absolute_error(y_test, e.predict(X_test))
    #print "got error %0.2f" % this_err
    print name, 'finished'
    errvals = np.append(errvals, this_err)

pos = np.arange(errvals.shape[0])
srt = np.argsort(errvals)
plt.figure(figsize=(7,5))
plt.bar(pos, errvals[srt], align='center')
plt.xticks(pos, ests_labels[srt])
plt.xlabel('Estimator')
plt.ylabel('Median Absolute Error')

plt.show()
