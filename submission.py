# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:52:12 2022

@author: sanjay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#----------------------------------------------------------------------------
# READ DATA 
df = pd.read_csv("D:/A/EduBridge/Capstone/data_sydney_cleaned.csv")

df.columns
# Data Exploration
df.head()
df.describe()
df['id'].count()
df.dtypes

df["host_is_superhost"].value_counts()
df["host_identity_verified"].value_counts()
df["property_type"].value_counts()
df["room_type"].value_counts()
df["accommodates"].value_counts()
df["bathrooms_text"].value_counts()
df["bedrooms"].value_counts()
df["has_availability"].value_counts()
df["review_scores_rating"].value_counts()
df["instant_bookable"].value_counts()

# outliers in price.
df.boxplot(column="price")

# Remove prices = 0 replace 0's with NaN, drop all NaNs
df['price'] = df['price'].replace(0, np.nan)
df = df.dropna(axis=0, how='any', subset=['price'])

# consider only prices below 500
df = df[df['price'] <= 500]
df.count()
df.describe()

# Column names
cols = df.columns.tolist()
cols

# Missing values - remove 
df.isna().sum()

df = df.dropna(axis=0, how='any', subset=cols)
print(df.shape)
df.count()

# Drop unwanted columns
df["host_response_rate"].value_counts()
df = df.drop('id', axis=1)
df = df.drop('host_id', axis=1)
df = df.drop('host_response_rate', axis=1)
df = df.drop('host_acceptance_rate', axis=1)
df = df.drop('last_review', axis=1)
df = df.drop('review_scores_location', axis=1)
df = df.drop('host_since', axis=1)
df = df.drop('bathrooms_text', axis=1)
df = df.drop('amenities', axis=1)

# Count property types.
df_property_type_statistic = df.groupby(
    ['property_type'])['price'].describe().reset_index()
df_property_type_statistic
remove_property_type=[]

# remove property types which are less than 20 
for i in range(0,len(df_property_type_statistic)) :
    if df_property_type_statistic["count"][i] < 15:
        remove_property_type.append(
            df_property_type_statistic["property_type"][i])
        
df = df[~df['property_type'].isin(remove_property_type)]


# Count room types.
df_room_type_statistic = df.groupby(
    ['room_type'])['price'].describe().reset_index()
df_room_type_statistic


# Correlation
cor = df.corr()

# Visulaisation
#----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(cor, annot=True, linewidths=.5, ax=ax)

cor["price"].sort_values(ascending = False)
attributes = ["price", "accommodates", "bedrooms", 
              "beds","review_scores_rating"]
pd.plotting.scatter_matrix(df[attributes], figsize=(12, 8))


# Scatter Plot showing price against accomodation
df.plot(kind="scatter", x="accommodates", y="price", alpha=0.1)

# Histogram - Prices 
fig, ax = plt.subplots(figsize=(9, 9))
sns.histplot(data=df, x="price",bins=50).set(
    title="Listing Price Distribution")


sns.barplot(x="price", y="property_type", data=df).set(
    title="Property_Type Price Distribution")

fig, ax = plt.subplots(figsize=(9, 9))
sns.barplot(x="count", y="property_type", data=df_property_type_statistic)

fig, ax = plt.subplots(figsize=(9, 9))
sns.barplot(x="count", y="room_type", data=df_room_type_statistic).set(
    title="Room Type Distribution")


# Encoding
#----------------------------------------------------------------------------

# Create X and y arrays
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# fuction to transfrom categorical columns.
def encode_label(arr_x, df_x):
    '''
    INPUT
    arr_x - X array
    df_x - dataframe
    
    OUTPUT
    arrays with numerical values
    '''
    le = LabelEncoder()
    index = 0
    for col in df_x.columns:
        if df_x[col].dtype == "object":
            arr_x[:,index] = le.fit_transform(arr_x[:,index])
            index = index + 1 
        else:
            index = index + 1      
    return arr_x


# label encoding - X
X = encode_label(X, df)


# label encoding - y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) 


# Split data
X_train, X_test, y_train, y_test =  train_test_split(
    X,y,test_size = 0.2, random_state= 0)


# Model 1

# Training the Multiple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#regressor.coef_

# test the prediction on test data
y_pred = regressor.predict(X_test)

# check the type of pred and test
y_test.dtype
y_pred.dtype

# convert pred to int
#y_test = y_test.astype(np.float64)
y_pred = y_pred.astype(np.int64)

print(np.concatenate((y_pred.reshape(
    len(y_pred),1), y_test.reshape(len(y_test),1)),1))


#Metrics 
#----------------------------------------------------------------------------


r2 = r2_score(y_test, y_pred)
print("R_Squared = ",r2)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error = ",rmse)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error = ",mae)

fig, ax = plt.subplots(figsize=(9, 9))
sns.regplot(x=y_test, y=y_pred, marker="+").set(title="Test")




# Recommend Price
#------------------------------------------------------------------------

dict_client = {}

#dict_client["id"] = 219030
#dict_client["host_id"] = 490291
dict_client["host_is_superhost"] = "t"
dict_client["host_listings_count"] = 2
dict_client["host_identity_verified"] = "t"
dict_client["latitude"] = -33.8008 
dict_client["longitude"] = 151.264
dict_client["property_type"] = "Entire residential home"
dict_client["room_type"] = "Entire home/apt"
dict_client["accommodates"] = 6
dict_client["bedrooms"] = 3
dict_client["beds"] = 3
dict_client["minimum_nights"] = 2
dict_client["maximum_nights"] = 22
dict_client["has_availability"] = "t"
dict_client["number_of_reviews"] = 3  
dict_client["review_scores_rating"] = 4.67
dict_client["instant_bookable"] = "t"



# dict_client2 = {}

# dict_client2["host_is_superhost"] = "t"
# dict_client2["host_listings_count"] = 3
# dict_client2["host_identity_verified"] = "t"
# dict_client2["latitude"] = -33.8796 
# dict_client2["longitude"] = 151.217
# dict_client2["property_type"] = "Private room in residential home"
# dict_client2["room_type"] = "Private room"
# dict_client2["accommodates"] = 4
# dict_client2["bedrooms"] = 2
# dict_client2["beds"] = 3
# dict_client2["minimum_nights"] = 1
# dict_client2["maximum_nights"] = 30
# dict_client2["has_availability"] = "t"
# dict_client2["number_of_reviews"] = 61  
# dict_client2["review_scores_rating"] = 4.95
# dict_client2["instant_bookable"] = "f"



def transform_client_array(client_array):  
    '''
    INPUT
    client_array - X array
    
    OUTPUT
    arrays with numerical values
    '''
    labelencoder_X = LabelEncoder()
    
    client_array[:,0] = labelencoder_X.fit_transform(client_array[:,0])
    client_array[:,2] = labelencoder_X.fit_transform(client_array[:,2])
    client_array[:,5] = labelencoder_X.fit_transform(client_array[:,5])
    client_array[:,6] = labelencoder_X.fit_transform(client_array[:,6])
    client_array[:,12] = labelencoder_X.fit_transform(client_array[:,12])
    client_array[:,15] = labelencoder_X.fit_transform(client_array[:,15])
    
    return client_array


def recommend_price(client_details):
    '''
    INPUT
    client_details - X array
    
    OUTPUT
    price prediction
    '''
    df_client = pd.DataFrame.from_dict(client_details, orient='index').T
    
    client_array = df_client.iloc[:,].values
    client_X = transform_client_array(client_array)
    
    rec_price = regressor.predict(client_X)
    
    return rec_price
    

recommended_price = recommend_price(dict_client)
print(recommended_price[0])
