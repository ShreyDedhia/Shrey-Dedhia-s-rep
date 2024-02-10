#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as pt
import random 
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
#import polars as pl
#from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
#import tensorflow.keras as keras
# from keras.models import Sequential
# from keras.layers import Dense
# from tensorflow.keras import metrics
# from sklearn.preprocessing import StandardScaler
# from keras.layers import Dropout


# In[7]:


data = pd.read_csv('train_data.csv',nrows = 1000000)
#data = pl.read_csv('train_data.csv')
labels  = pd.read_csv('train_labels.csv')


# In[8]:


#Step 2 
data['S_2'] = pd.to_datetime(data['S_2'])
data['year_month'] = data['S_2'].dt.to_period('M')
data["count"] = data.groupby(["year_month"])["customer_ID"].transform("count")
data["weight"] = (1/data["count"] )**3
data["weight"] = data["weight"]/data["weight"].unique().sum()
#data['S_2'] = data['S_2'].str_to_datetime(format="%Y-%m-%d")
data = data.sample(weights="weight",frac=1)
sample = data.groupby(['customer_ID'],as_index = False).first()
#Step 3
final_data = pd.merge(labels,sample, how = 'inner', on = 'customer_ID')


# In[59]:


final_data.to_csv('final_data.csv')


# In[9]:


final_data['year_month'].value_counts()


# In[10]:


#Step 4 
print(final_data.shape)
print(final_data.dtypes)
print(final_data.head())


# In[6]:


final_data = pd.read_csv('final_data.csv')
final_data.drop('Unnamed: 0',inplace = True,axis=1) # you can start from here to skip the first 4 steps.
final_data.drop(['count','weight'],inplace = True,axis=1) # drop count and weight


# In[11]:


# step 5
final_data_store = final_data.copy()
try:
    final_data.drop(['customer_ID'],inplace=True,axis=1)
    final_data.drop(['S_2'],axis=1,inplace = True)
except:
    print('already dropped')
cat_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
final_data[cat_cols] = final_data[cat_cols].astype('category')
df_numerical = final_data.drop(['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68'],axis=1)
onehot = OneHotEncoder(handle_unknown='ignore')
cat_cols_encoded = onehot.fit_transform(final_data[cat_cols])
cat_cols_encoded = cat_cols_encoded.toarray()
cat_cols_encoded_df = pd.DataFrame(cat_cols_encoded, columns=onehot.get_feature_names(cat_cols))
final_data_encoded = pd.concat([cat_cols_encoded_df.reset_index(drop=True), df_numerical.reset_index(drop=True)], axis=1)
final_data_encoded.fillna(np.nan)


# In[12]:


final_data_encoded.shape


# In[13]:


#Step 6
train  = final_data_encoded[(final_data_encoded['year_month']>='2017-05')&(final_data_encoded['year_month']<='2018-01')]
test1 = final_data_encoded[(final_data_encoded['year_month']>='2017-03')&(final_data_encoded['year_month']<='2017-04')]
test2 = final_data_encoded[(final_data_encoded['year_month']>='2018-02')&(final_data_encoded['year_month']<='2018-03')]


# In[22]:


#Step 7
try:
    train.drop(['year_month'],axis=1,inplace=True)
    test1.drop(['year_month'],axis=1,inplace=True)
    test2.drop(['year_month'],axis=1,inplace=True)
except:
    print('already dropped')
try:
    x = train.drop('target',axis=1)
    y = pd.DataFrame(train['target'])
except:
    print('already dropped')

model = XGBClassifier(random_state = 123)
model.fit(x,y)

importances = model.feature_importances_

# Get a list of feature names
feature_names = x.columns

# Create a dictionary of feature names and importance scores
feature_importances1 = dict(zip(feature_names, importances))

# Set a threshold for feature importance scores
threshold = 0.005

# Filter out features with importance scores below the threshold
important_features1 = [feature for feature, importance in feature_importances1.items() if importance > threshold]


# In[24]:


# Step 8
model1 =XGBClassifier(n_estimators=300, learning_rate=0.5, max_depth=4, subsample=0.5, 
                          colsample_bytree=0.5, min_child_weight=5, random_state=123)
model1.fit(x,y)
importances = model1.feature_importances_

feature_names = x.columns

feature_importances2 = dict(zip(feature_names, importances))

threshold = 0.005

important_features2 = [feature for feature, importance in feature_importances2.items() if importance > threshold]


# In[26]:


#Step 9
important_features1.extend(important_features2)
final_features = set(important_features1)
final_data_copy = final_data_encoded.copy()
final_data_copy = final_data_copy[final_features]
final_data_copy.to_csv('final_data_features.csv')
len(final_features)
# train = train[final_features]
# test1 = test1[final_features]
# test2 = test2[final_features]


# In[28]:


def get_xy(data):
    x = data.drop('target',axis=1)
    y = pd.DataFrame(data['target'])
    return(x,y)

x,y = get_xy(train)
x = x[final_features]
x_test1,y_test1 = get_xy(test1)
x_test2,y_test2 = get_xy(test2)
x_test1 = x_test1[final_features]
x_test2 = x_test2[final_features]


# In[49]:


#step 10 
scores = pd.DataFrame()
t =[]
l =[]
s =[]
c =[]
w =[]
atrain =[]
atest1 =[]
atest2 =[]
for trees in [50,100,300]:
    for LR in [0.01,0.1]:
        for subsample in [0.5,0.8]:
            for colsample in [0.5,1]:
                for weight in [1,5,10]:
                    model  = XGBClassifier(n_estimators=trees, learning_rate=LR, subsample=subsample, 
                              colsample_bytree=colsample, min_child_weight=weight, random_state=42)
                    model.fit(x,y)
                    t.append(trees)
                    l.append(LR)
                    s.append(subsample)
                    c.append(colsample)
                    w.append(weight)
                    temp = pd.DataFrame(model.predict_proba(x),columns=['prob of 0','prob of 1'])
                    temp = temp['prob of 1']
                    temp1 = pd.DataFrame(model.predict_proba(x_test1),columns=['prob of 0','prob of 1'])
                    temp1 = temp1['prob of 1']
                    temp2 = pd.DataFrame(model.predict_proba(x_test2),columns=['prob of 0','prob of 1'])
                    temp2 = temp2['prob of 1']
                    atrain.append(roc_auc_score(y, temp))
                    atest1.append(roc_auc_score(y_test1, temp1))
                    atest2.append(roc_auc_score(y_test2, temp2))
scores['trees'] = t
scores['Learning rate'] = l
scores['subsample'] = s
scores['percentage features'] = c
scores['Weight of default'] = w
scores['AUC train 1'] = atrain 
scores['AUC test 1'] = atest1
scores['AUC test 2'] = atest2
scores.to_csv('XGBoost_scores.csv')                  


# In[30]:


grid = {
    'n_estimators': [50, 100, 300],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.5, 1.0],
    'min_child_weight': [1, 5,10],
}


# In[31]:


model3 = XGBClassifier()
grid_search = GridSearchCV(estimator=model3, param_grid=grid, cv=5, scoring='roc_auc')
grid_search.fit(x, y)

# Print the best hyperparameters and their corresponding R^2 score
print('Best parameters:', grid_search.best_params_)
print('Best accuracy:',grid_search.best_score_)


# In[41]:


temp = pd.DataFrame(best_model.predict_proba(x),columns=['prob of 0','prob of 1'])
temp['prob of 1']


# In[48]:


#step 11
best_model = XGBClassifier(colsample_bytree =  0.5, learning_rate= 0.1, min_child_weight = 10, n_estimators =  100, subsample =  0.8)
best_model.fit(x,y)
temp = pd.DataFrame(best_model.predict_proba(x),columns=['prob of 0','prob of 1'])
temp = temp['prob of 1']
temp1 = pd.DataFrame(best_model.predict_proba(x_test1),columns=['prob of 0','prob of 1'])
temp1 = temp1['prob of 1']
temp2 = pd.DataFrame(best_model.predict_proba(x_test2),columns=['prob of 0','prob of 1'])
temp2 = temp2['prob of 1']

result_xgboost = pd.DataFrame({'Train_score': roc_auc_score(y, temp),
                               'Test1_Score': roc_auc_score(y_test1, temp1),
                               'Test2_Score': roc_auc_score(y_test2, temp2)},
                               index=['scores'])

result_xgboost = result_xgboost.T
result_xgboost


# In[24]:


x_test1.describe(percentiles = [0.01,0.99]).transpose()


# In[23]:


#Step 12
# fill missing values with 0
x = x.fillna(0)
x_test1 = x_test1.fillna(0)
x_test2 = x_test2.fillna(0)

# replace outliers with 1st and 99th percentiles
for feature in final_features:
    lb = np.percentile(x[feature], 1)
    ub = np.percentile(x[feature], 99)
    x.loc[x[feature] < lb, feature] = lb
    x.loc[x[feature] > ub, feature] = ub
    x_test1.loc[x_test1[feature] < lb, feature] = lb
    x_test1.loc[x_test1[feature] > ub, feature] = ub
    x_test2.loc[x_test2[feature] < lb, feature] = lb
    x_test2.loc[x_test2[feature] > ub, feature] = ub

# standardize the data
scaler = StandardScaler()
scaler.fit(x)
x_normalised = scaler.transform(x)
x_test1_normalised = scaler.transform(x_test1)
x_test2_normalised = scaler.transform(x_test2)

# create dataframes for the normalized data
x_normalised = pd.DataFrame(x_normalised, columns=x.columns)
x_test1_normalised = pd.DataFrame(x_test1_normalised, columns=x_test1.columns)
x_test2_normalised = pd.DataFrame(x_test2_normalised, columns=x_test2.columns)


# In[25]:


# Step 13
Neural = {}
hidden_layers = [2,4]
nodes = [4,6]
activation_func = ['relu','tanh']
dropout_rate = [0.5,0]
Batch_size = [100, 10000]
h =[]
n = []
a = []
d =[]
b =[]
atrain = []
atest1 = []
atest2 = []
for hl in hidden_layers:
    for nn in nodes:
        for acti in activation_func:
            for dr in dropout_rate:
                for bs in Batch_size:
                    model = Sequential()
                    model.add(Dense(nn, input_dim=len(final_features), activation=acti))
                    for i in range(hl - 1): # becasuse we have to set the last layer as the output node
                        model.add(Dense(nn, activation=acti))
                        if dr > 0:
                            model.add(Dropout(dr))
                    model.add(Dense(1, activation='sigmoid'))
                    
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.AUC()])
                    model1 = model.fit(x_normalised[final_features], y, epochs=20, batch_size=bs, verbose=0)
                    atrain.append(roc_auc_score(y, model.predict(x)))
                    atest1.append(roc_auc_score(y_test1, model.predict(x_test1)))
                    atest2.append(roc_auc_score(y_test2, model.predict(x_test2)))
                    h.append(hl)
                    n.append(nn)
                    a.append(acti)
                    d.append(dr)
                    b.append(bs)
    
    
Neural['hidden layers'] = h
Neural['nodes'] = n
Neural['activation function'] = a
Neural['dropout rate'] = d
Neural['batch size'] = b
Neural['AUC train 1'] = atrain 
Neural['AUC test 1'] = atest1
Neural['AUC test 2'] = atest2
Neural = pd.DataFrame(Neural)
Neural.to_csv('Neural_Netwoks.csv')


# In[71]:


def build_classifier(hidden_layers, nodes, activation_func, dropout_rate, optimizer,batch_size):
    final_nn_model = Sequential()
    final_nn_model.add(Dense(nodes, input_dim=len(final_features), activation=activation_func))
    for i in range(hidden_layers): # add hidden layers
        final_nn_model.add(Dense(nodes, activation=activation_func))
        if dropout_rate > 0:
            final_nn_model.add(Dropout(dropout_rate))
    final_nn_model.add(Dense(1, activation='sigmoid'))
    final_nn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[metrics.AUC()])
    final_nn_model.fit(x_normalised, y, epochs=20, batch_size=batch_size, verbose=0) # use batch_size in fit method
    return final_nn_model


keras_classifier = KerasClassifier(build_fn=build_classifier)

param_grid = {
    'hidden_layers': [2,4],
    'nodes': [4,6],
    'activation_func': ['relu', 'tanh'],
    'dropout_rate': [0.5,0],
    'batch_size':[100, 10000],
    'optimizer': ['adam']
}

grid_search = GridSearchCV(estimator=keras_classifier, param_grid=param_grid, cv=5)
grid_search_result = grid_search.fit(x_normalised, y)

print(f"Best AUC Score: {grid_search_result.best_score_:.4f}")
print(f"Best Parameters: {grid_search_result.best_params_}")


# In[26]:


# Step 14
final_nn_model = Sequential()
final_nn_model.add(Dense(6, input_dim=len(final_features), activation='tanh'))
for i in range(3): # 4 hidden layers
    final_nn_model.add(Dense(6, activation='tanh'))
    final_nn_model.add(Dropout(0)) # 0 dropout rate
final_nn_model.add(Dense(1, activation='sigmoid'))
final_nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.AUC()])
final_nn_model.fit(x_normalised, y, epochs=20, batch_size=100, verbose=0)
roc_auc_score(y,final_nn_model.predict(x_normalised))
roc_auc_score(y_test1,final_nn_model.predict(x_test1_normalised))
roc_auc_score(y_test2,final_nn_model.predict(x_test2_normalised))


# In[27]:


# Step 15
result_neural_networks = pd.DataFrame({'Train_score': roc_auc_score(y, final_nn_model.predict(x_normalised)),
                               'Test1_Score': roc_auc_score(y_test1, final_nn_model.predict(x_test1_normalised)),
                               'Test2_Score': roc_auc_score(y_test2, final_nn_model.predict(x_test2_normalised))},
                               index=['scores'])
result_neural_network = result_neural_networks.T
result_neural_network
# by comparing the results of both the models. we can conclude that the neural networks is a better choice model. 


# In[96]:


result_neural_network


# In[28]:


from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test2, final_nn_model.predict(x_test2_normalised))
auc_score = roc_auc_score(y_test2, final_nn_model.predict(x_test2_normalised))
pt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
pt.xlabel('False Positive Rate')
pt.ylabel('True Positive Rate')
pt.title('Receiver operating characteristic (ROC) curve')
pt.legend(loc="lower right")


# In[100]:


best_threshold = thresholds[np.argmax(tpr-fpr)] 
best_threshold
# the optimum threshold is 0.30300188
# we do not want to loose the customer to we maximize the true positive rate


# In[30]:


result = pd.DataFrame(final_nn_model.predict(x_test2_normalised),columns=['probability of default'])


# In[44]:


response_rate = []
t=[]
for i in range(1,100):
    result['prediction'] = result['probability of default'] > i/100
    result['prediction'] = result['prediction'].astype(int)
    response_rate.append(result['prediction'].mean()*100)
    t.append(i/100)
response_rate = pd.DataFrame({'Threshold':t,'Response_Rate':response_rate})
response_rate = response_rate[response_rate['Response_Rate']<10]
response_rate
# the ideal response rate is 0.87


# In[46]:


pt.plot(response_rate)


# In[150]:





# In[48]:


# slide 2
val = {}
year = []
rates = []
for year_month in final_data['year_month'].unique():
    rate = final_data[final_data['year_month'] == year_month]['target'].mean()*100
    rates.append(rate)
    year.append(year_month)
val['year_month'] = year
val['rate'] = rates
val = pd.DataFrame(val)
counts = final_data['year_month'].value_counts().reset_index()
counts.columns = ['year_month', 'count']
merged_data = pd.merge(val, counts, on='year_month', how='inner')
merged_data.to_csv('slide2.csv',index = False)


# In[49]:


# Slide 3
col= data[['P_2','D_88','D_132','D_42','D_53']]
stats = col.describe(percentiles=[0.01,0.05,0.95,0.99],).transpose()
percentage_missing  = ((len(col) - stats['count']) / len(col)) * 100
stats['% of missing'] = percentage_missing
stats.to_csv('slide3.csv',index=False)


# In[73]:


import seaborn as sns
my_palette = sns.color_palette([(1, 1, 1), (0.2, 0.2, 0.7)])
feature_importances_df = pd.DataFrame(feature_importances2.items(), columns=['Feature', 'Importance'])
pt.scatter(feature_importances_df['Feature'],feature_importances_df['Importance'])
pt.xlabel('Importance')
pt.ylabel('Feature')
pt.title('Feature Importance')


# In[77]:


feature_importances_df = pd.DataFrame.from_dict(feature_importances2, orient='index', columns=['Importance'])
feature_importances_df.index.name = 'Feature'
feature_importances_df.sort_values(by='Importance', ascending=False, inplace=True)


# In[87]:





# In[101]:


#Slide6
list_final_features = {key: (feature_importances1[key],feature_importances2[key]) for key in feature_importances1.keys()&feature_importances2.keys() if feature_importances1[key]>0.005 or feature_importances2[key]>0.005}
for key,value in list_final_features.items():
    list_final_features[key] = max(value)
len(list_final_features)
pt.barh(list(list_final_features.keys()),list(list_final_features.values()))
pt.yticks([0, 10, 20, 30, 40])
pt.show()


# In[122]:


#slide8 
#plot 1
temp = scores.iloc[:,-3:]
pt.scatter(np.mean(temp,axis=1),np.std(temp,axis=1))
pt.xlabel('Average AUC')
pt.ylabel('Standard Deviation of AUC')


# In[123]:


# plot 2 
pt.scatter(scores['AUC train 1'],scores['AUC test 2'])
pt.xlabel('Train AUC')
pt.ylabel('Test 2 AUC')


# In[141]:


# slide 10 
import seaborn as sns
sns.stripplot(x_test2_normalised['D_41'],y_test2.values)


# In[162]:


#Slide 10
import shap
explain  = shap.TreeExplainer(best_model)
shap_values = explain.shap_values(x_test2)
#expected_value  = shap_values[0,-1]
#shap_values = shap_values[::-1]
#shap.initjs()
shap.summary_plot(shap_values,x_test2)


# In[169]:


# Slide 11
explainer = shap.Explainer(best_model)
observation = x_test2
shap_values = explainer(observation)
shap.waterfall_plot(shap_values[1], max_display=10, show=False)


# In[167]:


observation = x_test2.iloc[1]
shap_values = explainer(observation)


# In[168]:


x_test2.iloc[1]


# In[170]:


#slide 14
#plot 1
temp = Neural.iloc[:,-3:]
pt.scatter(np.mean(temp,axis=1),np.std(temp,axis=1))
pt.xlabel('Average AUC')
pt.ylabel('Standard Deviation of AUC')


# In[171]:


# plot 2 
pt.scatter(Neural['AUC train 1'],Neural['AUC test 2'])
pt.xlabel('Train AUC')
pt.ylabel('Test 2 AUC')


# In[172]:





# In[ ]:




