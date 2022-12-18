#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 필요한 라이브러리 호출
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[99]:


# 필요한 데이터 titanic 호출
titanic = pd.read_csv('C:/myPyCode/data/train.csv')
titanic


# In[8]:


titanic.head()


# In[10]:


titanic.info()#object : Name, Sex, Ticket, Cabin, Embarked 처리


# In[11]:


titanic.describe()


# In[12]:


#상관관계
titanic.corr()


# In[92]:


# 상관관계 시각화+수치
sns.heatmap(titanic.corr(), vmax = 1, vmin = -1, annot=True, cmap = 'crest')


# In[14]:


# 종속변수 Pclass로 결정, 독립변수는 그외 나머지


# In[15]:


# 불필요한 PassengerId 삭제
titanic.set_index('PassengerId', inplace = True)


# In[16]:


titanic.info()


# In[18]:


# 데이터 전처리 과정
titanic.isnull()


# In[19]:


titanic.isnull().sum()


# In[23]:


# age, cabin nan 값 조절
titanic.isnull().sum()/len(titanic)


# In[31]:


titanic.Embarked.value_counts().plot(kind='barh')


# In[33]:


# mebarked nan을 s로 편입

titanic.Embarked = titanic.Embarked.fillna('S')
titanic.Embarked


# In[34]:


titanic.isnull().sum()


# In[ ]:


titanic.set_index('Name', inplace=True)
# Name 컬럼 삭제


# In[37]:


titanic.info()


# In[49]:


titanic.set_index('PassengerId', inplace = True) #Cabin 컬럼 삭제


# In[39]:


titanic.isnull().sum() # age가 문제


# In[50]:


titanic.isnull().sum().plot(kind='barh')


# In[48]:


titanic


# In[51]:


# age 결측치는 평균으로 통일
titanic.Age = titanic.Age.fillna(titanic.Age.mean())
titanic.Age


# In[53]:


titanic.info()


# In[59]:


titanic.index = np.arange(891)
titanic


# In[60]:


#ticket 컬럼 드랍
titanic.drop('Ticket', axis = 1, inplace=True)
titanic.info()


# In[61]:


# 더미데이터 생성 -sex, Embarked 열

titanic = pd.get_dummies(titanic, columns=['Sex','Embarked'], drop_first = True)


# In[62]:


titanic


# In[66]:


# 훈련셋 , 시험셋 분리 - 라이브러리 호출
from sklearn.model_selection import train_test_split


# In[67]:


# 독립변수, 종속변수 지정 (pclass 제외 모두가 X)
X = titanic.drop('Pclass', axis = 1)
y = titanic['Pclass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)


# In[69]:


X_train
X_test
y_train
y_test


# In[72]:


# 분류문제 학습
from sklearn.linear_model import LogisticRegression


model = LogisticRegression()
model.fit(X_train, y_train)


# In[74]:


# 최적화 완료
# 시험셋 예측 시작

pred = model.predict(X_test)
pred


# In[75]:


# proba로 각 클래스 예측확률 모의계산

pred_proba = model.predict_proba(X_test)
pred_proba


# In[76]:


# 모델 평가 sklearn.metrics에서 accuracy_score 호출

from sklearn.metrics import accuracy_score


# In[78]:


score = accuracy_score(y_test, pred)
score


# In[80]:


# 기울기 확인
model.coef_


# In[81]:


# 절편 확인
model.intercept_


# In[82]:


# 결정계수 확인
model.score(X_train, y_train)


# In[89]:


# 기울기 차원 축소
model.coef_[0].shape


# In[90]:


# 차원 축소한 기울기로 column 별 지표를 series화
pd.Series(model.coef_[0], index = X.columns)

