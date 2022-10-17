"""
Author: Alex Yang
This module is about building up a model to predict flood occurrence. 
The knowledge behind the model constructed is to take the normal monthly weather data to subtract the actual observed monthly weather data to get the difference, in this case, we can get a sense of how huge the difference in every variable is correlated with the occurrence of the flood. 
The KNN method is used to construct the model after dealing with class imbalance. 
The model is created for predicting the probability of flood occurrence. 
For future use, the government can simply use the data of detected precipitation, pressure level, and temperature to predict whether the difference compared to what happened normally is huge and might see floods happening again. 
In this way, the government can alert their residents and have some precautions beforehand by tracking the result of plugging real-time into the flood predicting model.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter

path = 'weather.csv'

#import organized data and get another dataframe which consists of differences in the parameters in two groups of data  
def get_data():
    df = pd.read_csv(path)
    return df

def create_df_diff():
    df = get_data()
    df['Floods'] = df['Floods'].fillna(0).astype(int)
    df_normal = df[df['Observed_Normal'] == 'normal']
    df_observed = df[df['Observed_Normal'] == 'observed']
    # Get the difference of all independent variables
    df_diff = df_observed.iloc[:,2:-1].reset_index().subtract(df_normal.iloc[:,2:-1].reset_index()) 
    return df_diff

# Check if class imbalance is true
def check_class_balancement(df_diff):
    sns.distplot(df_diff['Floods'])
    counter = Counter(df_diff.Floods)
    for k,v in counter.items():
        per = v/len(df_diff.Floods) *100
        print('Class=%s, Count=%d, Percentage=%.2f%%' % (k, v, per))

# Use random over-sampling to deal with class imbalance
def rand_over_samp(df_diff):
    # Since the original dataset doesn't include too many rows, the usage of over-sampling can avoid subsetting rows
    class_1_over_samp = df_diff[df_diff['Floods'] == 1].sample(len(df_diff[df_diff['Floods'] == 0]), replace=True)
    test_over = pd.concat([class_1_over_samp, df_diff[df_diff['Floods'] == 0]], axis=0)

    return test_over

def visual_rand_over_samp():
      # %% 
    print("total class of 1 and 0:",test_over['Floods'].value_counts())# plot the count after under-sampeling
    test_over['Floods'].value_counts().plot(kind='bar', title='count (Floods)')

# Importing packages for the model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Splitting data
def data_splitting(test_over):
    x = test_over.iloc[:, 2:-1].values
    y = test_over.iloc[:, 1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0, stratify=y)
    # Make sure of the shape of the data 
    print("x_train.shape: ", x_train.shape,"x_test.shape: ", x_test.shape, "y_train.shape: ", y_train.shape, "y_test.shape: ", y_test.shape)
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.fit_transform(x_test)
    return x_train,x_test,y_train,y_test

# Scaling and Fitting Data
def create_model(x_train,y_train):
    # set the weights to distance to reduces the effect of a skewed representation of data
    clf = KNeighborsClassifier(weights='distance')
    clf.fit(x_train, y_train)
    return clf

def result():
    y_pred = model.predict(x_test)

    # Model Preview
    cm = confusion_matrix(y_pred, y_test)
    print("Confusion Metrix: ", cm)

    # Viewing the actual prediction
    res = pd.DataFrame({"Actual": y_test, 'Predicted': y_pred})
    print(res)

    # Model Result
    print(f"F1: {np.mean(f1_score(y_test,y_pred)).round(2)}") 
    #F1: 0.85
    print(f"Accuracy: {np.mean(accuracy_score(y_test,y_pred)).round(2)}") 
    #Accuracy: 0.82
    print(classification_report(y_test ,y_pred))
    print(model.score(x_test,y_test))
    #model score: 0.824561403508771


df_diff = create_df_diff()
test_over = rand_over_samp(df_diff)
#visual_rand_over_samp(test_over)
x_train,x_test,y_train,y_test = data_splitting(test_over)
model = create_model(x_train,y_train)
result()