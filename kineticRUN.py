
# coding: utf-8

# In[1]:

import  pandas as pd
import numpy as np


# In[2]:

train=pd.read_csv('/media/machine_learning/A80C461E0C45E7C01/dogs/kineticTrainGOOD.csv')


# In[11]:

test=pd.read_csv('/media/machine_learning/A80C461E0C45E7C01/dogs/kineticTestGOOD.csv)


# In[4]:

y_train=train['interest_level']


# In[5]:

train=train.drop('interest_level',axis=1)
x_train=np.array(train)


# In[6]:

x_test=np.array(test)


# In[7]:

import  xgboost  as xb


# In[8]:

# get  id 
test_file = '/media/machine_learning/A80C461E0C45E7C01/dogs/test.json'
test1 = pd.read_json(test_file)
listing_id = test1.listing_id.values


# In[14]:

#########################################################################################################################
###########until here I have    new processed train and test  and  must  hstack  with  NLP  csr matrix##################
#import  xgboost  as xgb
#clf=xgb.XGBClassifier(n_estimators=3000)
#clf.fit(x_train,y_train)


# In[15]:

train_test=pd.concat((train,test),axis=0).reset_index(drop=True)
temp=train_test
train_test1=train_test


# In[16]:

#read again  things in order for  sparse matrixes
train_file = '/media/machine_learning/A80C461E0C45E7C01/dogs/train.json'
test_file = '/media/machine_learning/A80C461E0C45E7C01/dogs/test.json'
train = pd.read_json(train_file)
test = pd.read_json(test_file)
listing_id = test.listing_id.values
y_map = {'low': 2, 'medium': 1, 'high': 0}
train['interest_level']=train['interest_level'].apply(lambda x:y_map[x])
y_train=train['interest_level'].values
train=train.drop(['interest_level','listing_id'],axis=1)
test=test.drop('listing_id',axis=1)
nrain=train.shape[0]
train_test=pd.concat((train,test),axis=0).reset_index(drop=True)


# In[46]:

#doing  like in script  processing for  text for columns features
ntrain=train.shape[0]


# In[20]:

train.iloc[300]['features']


# In[21]:

s=train.iloc[300]['features']


# In[22]:

' '.join(s)


# In[23]:

#whe  apply  lambda functional programming  in order  to   get  a string  composed with all words


# In[24]:

train_test['features2']=train_test['features']


# In[25]:

train_test['features2']=train_test['features2'].apply(lambda x:' '.join(x))


# In[26]:

train_test.iloc[300]['features2']


# In[27]:

#ok


# In[29]:

from sklearn.feature_extraction.text import  CountVectorizer
 
from scipy import sparse
c_vect = CountVectorizer(stop_words='english', max_features=200, ngram_range=(1, 1))
c_vect.fit(train_test['features2'])


# In[30]:

c_vect


# In[31]:


c_vect_sparse_1 = c_vect.transform(train_test['features2'])


# In[32]:

c_vect_sparse_1


# In[33]:

#a sparse matrix   with 200 columns  where  it contains  0 or  1  if a   certain row  correponds to the count vectorizer
#resulted  word


# In[34]:

c_vect_sparse1_cols = c_vect.get_feature_names()
len(c_vect_sparse1_cols)


# In[36]:

features = list(train_test1.columns)


# In[41]:

train_test_cv1_sparse = sparse.hstack((train_test1, c_vect_sparse_1)).tocsr()


# In[42]:

train_test_cv1_sparse


# In[47]:


x_train = train_test_cv1_sparse[:ntrain, :]
x_test = train_test_cv1_sparse[ntrain:, :]
features += c_vect_sparse1_cols


# In[53]:

SEED = 777
NFOLDS = 5
import xgboost as xgb
params = {
    'eta':.01,
    'colsample_bytree':.8,
    'subsample':.8,
    'seed':0,
    'nthread':16,
    'objective':'multi:softprob',
    'eval_metric':'mlogloss',
    'num_class':3,
    'silent':1
}


dtrain = xgb.DMatrix(data=x_train, label=y_train)
dtest = xgb.DMatrix(data=x_test)


bst = xgb.cv(params, dtrain, 10000, NFOLDS, early_stopping_rounds=50)
best_rounds = np.argmin(bst['test-mlogloss-mean'])

bst = xgb.train(params, dtrain, best_rounds)
preds = bst.predict(dtest)

preds = pd.DataFrame(preds)

cols = ['high', 'medium', 'low']

preds.columns = cols

preds['listing_id'] = listing_id

preds.to_csv('sparse_preds.csv', index=None)


# In[ ]:



