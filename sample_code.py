#!/usr/bin/env python
# coding: utf-8

# #### Import modules

# In[1]:


# Python 3.9.13, Conda 22.9.0


# In[2]:


import pandas as pd # version 1.4.4
import numpy as np # version 1.21.5
import matplotlib.pyplot as plt
import imblearn # version 0.9.1
import sklearn #version 1.1.3
import xgboost # version 1.7.1
import matplotlib # version 3.5.2
import seaborn as sns # version 0.11.2

import itertools
 
from sklearn.linear_model import LogisticRegression ### Use a logistic regression but change the threshold ... This often solves the problem
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score,accuracy_score,f1_score,recall_score,roc_auc_score # F1 score will probably be the most important one
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE


# In[3]:


import warnings
warnings.filterwarnings("ignore") 


# #### Not using neural networks because its not explainable.

# #### Read in datasets

# In[4]:


train = pd.read_csv(r"C:\Users\rodge\Downloads\Training_set.csv")
test = pd.read_csv(r"C:\Users\rodge\Downloads\Test_set.csv" )


# #### sanity checks for features derived from other features
# - codebook possible typo: in describing the variable "num_acc_30d_past_due_6_months" : Numer of accounts that are 30 or more days delinquent within last **62** months. There is no time dimension so we can't verify this with data. I think it should be 6 months.

# In[5]:


train.columns
train.columns


# In[6]:


sns.displot(train , x=train['rep_income'])


# In[7]:


train[train.tot_balance<train.avg_bal_cards]


# In[8]:


test[test.tot_balance<test.avg_bal_cards]


# In[9]:


train[train.num_acc_30d_past_due_12_months<train.num_acc_30d_past_due_6_months]


# In[10]:


test[test.num_acc_30d_past_due_12_months<test.num_acc_30d_past_due_6_months]


# #### Group by's

# In[11]:


train.groupby(["ind_XYZ"])["Def_ind"].value_counts(normalize=True) # people who dont have accounts with XYZ have a higher default rate


# In[12]:


train.groupby(["Def_ind"])["rep_education"].value_counts(normalize=True) 


# In[13]:


train.Def_ind.value_counts(normalize = True)


# In[14]:


train.groupby("rep_education")["rep_education"].count()


# #### Analyzing datasets

# In[15]:


def data_summary(df:pd.DataFrame):
    """Summarize the dataset: dimensions, columns present, summary statistics, data types
    class balance, rows and columns with Nans
    """
    print(f"Data Dimensions, rows: {df.shape[0]}, columns:{df.shape[1]}")
    print()
    print("Columns :\n",df.columns)
    print()
    print("Summary Statistics :\n",pd.DataFrame(df.describe()))
    print()
    print("Data Types:\n ",df.dtypes)
    print("columns with nans :\n",df.columns[df.isna().any()].tolist())
    print("percentage of rows with nans :\n", len(df[df.isna().any(axis=1)])/len(df))
    print("class balance :\n", df["Def_ind"].value_counts(normalize = True))
    


# #### Call function on the testing and training set

# In[16]:


#data_summary(train)
#data_summary(test)


# In[17]:


all_data = train.append(test )


# In[18]:


test[test.isna().any(axis=1)]


# #### Generate table of summary statistics

# In[19]:


all_data.describe().T.to_excel(r"Downloads\sum_stat.xlsx") ### this is the summary statistics table in the technical report.


# #### correlation :useful incase we were in a regression setting.

# In[20]:


plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(all_data.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')


# -Training set
#    - Data has 20000 rows and 21 columns
#    - 3 columns have Nan values, about 16% of the rows have a Nan
#    - Class balance: 10% have defaulted, 90% have not defaulted. This is an imbalanced dataset. We will need to think about how we handle this.

# -Testing set
#  - 5000 rows, 21 columns as expected.
#  - 3 columns have nans, about 17% of the rows have nan values
#  - Same class imbalance as the training set.

# ### Modelling

# - Before we do anything else we need to figure out how to handle the Nan values and the "rep_education" text column. The algorithms will throw an error with a str datatype
# - As a first pass, we will drop all rows with nan values
# - At this point we are also not creating any features yet. We implement a quick and dirty solution and the we iteratively improve it.

# #### Sparsity?

# ##### How sparse is the data?  Could be problemamtic with bigger matrices

# In[21]:


print('columns')
print(train[train == 0].count(axis=0)/len(train.index))


# In[22]:


print('columns')
print(test[test == 0].count(axis=0)/len(test.index))


# #### Fill in all nans with mode, mode, median as a first pass and One Hot Encode categorical text feature ("rep_education")

# In[23]:


cols = ['pct_card_over_50_uti', 'rep_income', 'rep_education']
test[cols]=test[cols].fillna(test.mode().iloc[0])
train[cols]=train[cols].fillna(train.mode().iloc[0])


# #### training set

# In[24]:


enc = OneHotEncoder(handle_unknown="ignore")
encode_df_train = pd.DataFrame(enc.fit_transform(train[['rep_education']]).toarray())
# rename columns to original columns.

cat_column_names = [item for items in enc.categories_ for item in items]
encode_df_train.columns = cat_column_names


# #### testing set

# In[25]:


enc = OneHotEncoder(handle_unknown="ignore")
encode_df_test = pd.DataFrame(enc.fit_transform(test[['rep_education']]).toarray())
# rename columns to original columns.

cat_column_names = [item for items in enc.categories_ for item in items]
encode_df_test.columns = cat_column_names


# #### Add back encoded columns \& assign to self

# In[26]:


train = pd.concat([train,encode_df_train], axis = 1)


# In[27]:


test = pd.concat([test,encode_df_test], axis = 1)


# #### drop text column

# In[28]:


train.drop(columns = "rep_education", inplace = True)
test.drop(columns = "rep_education", inplace = True)


# #### Separate features and labels

# In[29]:


train_X = train.loc[:, train.columns != "Def_ind"]
test_X = test.loc[:, test.columns != "Def_ind"]
train_y = train["Def_ind"]
test_y = test["Def_ind"]


# #### compare models quickly to select which one to use in addition to Logistic regression

# In[30]:


def train_compare_models(train_X: pd.DataFrame, train_y:pd.Series, test_X:pd.DataFrame, test_y:pd.Series)->pd. DataFrame:
    """ Function to train and compare multiple models quickly then improve
    :param train_x: training set variables (features)
    :param train_y: classes/labels for the training set
    :param test_x: test set variables (features)
    :param test_y: classes/labels for the test set
    """
    
    dataframes = []
    results = []
    algorithms = []
    
    models = [('LR', LogisticRegression()), ('RF', RandomForestClassifier()), ('XGB', XGBClassifier())]
    
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    target_names = ["did_not_default","defaulted"]
    
    for algorithm, model in models:
        kfold = model_selection.KFold(n_splits = 5, shuffle = True, random_state = 5)
        cv_results = model_selection.cross_validate(model, train_X, train_y, cv = kfold,  scoring =  scoring)
        clf = model.fit(train_X, train_y)
        pred_y = clf.predict(test_X)
        print(algorithm)
        print(classification_report(test_y,pred_y, target_names=target_names))
              
        results.append(cv_results)
        algorithms.append(algorithms)
              
        results_df = pd.DataFrame(cv_results)
        results_df['model'] = algorithm
        dataframes.append(results_df)
              
    all_results_df = pd.concat(dataframes, ignore_index=True)
    
    return all_results_df


# #### mode imputed data results

# In[31]:


data = train_compare_models(train_X, train_y, test_X, test_y)


# ## let's try another configuration where we drop all the nan's

# In[32]:


train = pd.read_csv(r"C:\Users\rodge\Downloads\Training_set.csv")
test = pd.read_csv(r"C:\Users\rodge\Downloads\Test_set.csv" )


# In[33]:


train_no_nan = train.dropna().reset_index()
test_no_nan = test.dropna().reset_index()


# #### one hot encode

# In[34]:


train_no_nan.drop(columns = "index", inplace =True)
test_no_nan.drop(columns = "index", inplace =True)


# In[35]:


enc = OneHotEncoder(handle_unknown="ignore")
encode_df_train = pd.DataFrame(enc.fit_transform(train_no_nan[['rep_education']]).toarray())
# rename columns to original columns.

cat_column_names = [item for items in enc.categories_ for item in items]
encode_df_train.columns = cat_column_names


# In[36]:


enc = OneHotEncoder(handle_unknown="ignore")
encode_df_test = pd.DataFrame(enc.fit_transform(test_no_nan[['rep_education']]).toarray())
# rename columns to original columns.

cat_column_names = [item for items in enc.categories_ for item in items]
encode_df_test.columns = cat_column_names


# In[37]:


train_no_nan.drop(columns = "rep_education", inplace = True)
test_no_nan.drop(columns = "rep_education", inplace = True)


# In[38]:


print(len(train_no_nan), len(encode_df_train),len(test_no_nan),len(encode_df_test))


# In[39]:


training_set = pd.concat([train_no_nan,encode_df_train], axis = 1)
testing_set = pd.concat([test_no_nan,encode_df_test], axis = 1)


# In[40]:


train_X = training_set.loc[:, training_set.columns != "Def_ind"]
test_X = testing_set.loc[:, testing_set.columns != "Def_ind"]
train_y = training_set["Def_ind"]
test_y = testing_set["Def_ind"]


# In[41]:


def train_compare_models(train_X: pd.DataFrame, train_y:pd.Series, test_X:pd.DataFrame, test_y:pd.Series)->pd. DataFrame:
    """ Function to train and compare multiple models quickly then improve
    :param train_x: training set variables (features)
    :param train_y: classes/labels for the training set
    :param test_x: test set variables (features)
    :param test_y: classes/labels for the test set
    """
    
    dataframes = []
    results = []
    algorithms = []
    
    models = [('LR', LogisticRegression()), ('RF', RandomForestClassifier()), ('XGB', XGBClassifier())]
    
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    target_names = ["defaulted", "did_not_default"]
    
    for algorithm, model in models:
        kfold = model_selection.KFold(n_splits = 5, shuffle = True, random_state = 5)
        cv_results = model_selection.cross_validate(model, train_X, train_y, cv = kfold,  scoring =  scoring)
        clf = model.fit(train_X, train_y)
        pred_y = clf.predict(test_X)
        print(algorithm)
        print(classification_report(test_y,pred_y, target_names=target_names))
              
        results.append(cv_results)
        algorithms.append(algorithms)
              
        results_df = pd.DataFrame(cv_results)
        results_df['model'] = algorithm
        dataframes.append(results_df)
              
    all_results_df = pd.concat(dataframes, ignore_index=True)
    
    return all_results_df


# In[42]:


data_no_nan = train_compare_models(train_X, train_y, test_X, test_y)


# #### The classification report reveals two things.
#  - The performance of the algorithms on the minority class is subpar
#  - And surprisingly the Random forest outperforms the XGBoost and has a slower training time

# #### Let's tweak the parameters of the Logistic regression and the random forest and check if we can improve their performance

# #### RF hyperparameters

# #### We will use a parameter grid to tune a hyper parameters of the random forest

# In[43]:


train_X = training_set.loc[:, training_set.columns != "Def_ind"]
test_X = testing_set.loc[:, testing_set.columns != "Def_ind"]
train_y = training_set["Def_ind"]
test_y = testing_set["Def_ind"]


# In[44]:


param_grid = {
    "n_estimators":[100,500,1000],
    "max_depth":[10, 50, 100],
    "max_features":[6,14,16,24 ]
}


# In[45]:


rf = RandomForestClassifier()


# In[46]:


rf_optimized = GridSearchCV(estimator=rf,
                            param_grid=param_grid,
                            cv=3,
                            n_jobs=-1,
                            verbose=2)


# In[47]:


rf_optimized.fit(train_X, train_y)
rf_optimized.best_estimator_ ### best model max_depth = 100, max_features = 14, n_estimators = 1000


# #### Lets see feature importances for the best model

# In[48]:


rf_best = rf_optimized.best_estimator_


# In[49]:


col_names = train_X.columns


# In[50]:


f = pd.DataFrame(rf_best.feature_importances_, index=col_names, columns=["feature_importance"])
f.sort_values("feature_importance", ascending=False)


# #### fitting model with best estimator above.

# In[51]:


rf_best = RandomForestClassifier(max_depth=100, max_features=14, n_estimators=1000)


# In[52]:


model = rf_best.fit(train_X, train_y)


# In[53]:


pred_y = model.predict(test_X)


# In[54]:


target_names = ["did_not_default","defaulted"]
print("RF")
print(classification_report(test_y,pred_y, target_names=target_names))


# In[55]:


cm = confusion_matrix(test_y,pred_y, labels=[0,1])


# In[56]:


fig = plt.figure()
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax,cmap='Greens');  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels([ "did_not_default","defaulted"]); ax.yaxis.set_ticklabels([ "did_not_default","defaulted"]);
fig.savefig('cm_rf.png')


# In[57]:


roc_auc_score(test_y,pred_y)


# #### Logistic regression hyperparameters

# In[58]:


pipe = Pipeline([('classifier' ,  LogisticRegression())])


# In[59]:


param_grid = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l1', 'l2'],
    'classifier__C' : np.logspace(-4, 4, 20),
    'classifier__solver' : ['liblinear']}]


# In[60]:


lr = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)


# In[61]:


lr.fit(train_X, train_y)
lr.best_estimator_ 


# In[62]:


lr_best = LogisticRegression(C=0.08858667904100823, penalty='l1', solver='liblinear')


# In[63]:


lr_best = lr_best.fit(train_X, train_y)


# #### We make considerations here for the use case. Because this is a client facing application, we want to reject applications if and only iff there is evidence that the client will default. We achieve this by setting an abitrarily high cutoff for the prediction. We will use the predict_proba instead of predict for this case

# In[64]:


set_threshold = 0.5


# In[65]:


prediction = np.where(lr_best.predict_proba(test_X)[:,1] > set_threshold, 1, 0)


# In[66]:


pd.DataFrame(data=[accuracy_score(test_y, prediction), recall_score(test_y, prediction),
                   precision_score(test_y, prediction), roc_auc_score(test_y, prediction), f1_score(test_y, prediction)], 
             index=["accuracy", "recall", "precision", "roc_auc_score", "f1_score"], columns =["metrics"])


# In[67]:


print("LR")
print(classification_report(test_y,prediction, target_names=target_names)) # threshold 0.9


# In[68]:


set_threshold = 0.5 # with the same threshold


# In[69]:


prediction = np.where(lr_best.predict_proba(test_X)[:,1] > set_threshold, 1, 0)


# In[70]:


pd.DataFrame(data=[accuracy_score(test_y, prediction), recall_score(test_y, prediction),
                   precision_score(test_y, prediction), roc_auc_score(test_y, prediction), f1_score(test_y, prediction)], 
             index=["accuracy", "recall", "precision", "roc_auc_score", "f1_score"], columns =["metrics"])


# In[71]:


print(classification_report(test_y,prediction, target_names=target_names))


# In[72]:


cm = confusion_matrix(test_y,prediction, labels=[0,1])


# In[73]:


fig = plt.figure()
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax,cmap='Greens');  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels([ "did_not_default","defaulted"]); ax.yaxis.set_ticklabels([ "did_not_default","defaulted"]);
fig.savefig('cm_lr.png')


# #### The logistic regression does not outperform the random forest on defaulters using the f1_score

# In[74]:


feature_importance = abs(lr_best.coef_[0])


# In[75]:


np.sort(feature_importance)


# In[76]:


feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5


# In[77]:


sorted_idx


# In[78]:


np.array(train_X.columns)[sorted_idx]


# In[79]:


featfig = plt.figure()
featax = featfig.add_subplot(1, 1, 1)
featax.barh(pos, feature_importance[sorted_idx], align='center')
featax.set_yticks(pos)
featax.set_yticklabels(np.array(train_X.columns)[sorted_idx], fontsize=8)
featax.set_xlabel('Relative Feature Importance')

plt.tight_layout()   
plt.show()


# In[ ]:




