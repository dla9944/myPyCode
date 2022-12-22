#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 필요한 라이브러리 호출

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn


# In[3]:


# 파일 로드
df = pd.read_csv('C:/myPyCode/data/car.csv')
df


# In[4]:


# 데이터 확인중
df.head()


# In[6]:


# object 숫자 확인
df.info() # name, fuel, seller_type, transmission, owner, mileage, engine,
# max_power, torque, 이상 확인


# In[8]:


# 소수점 축약
pd.options.display.float_format = "{:.4f}".format
df.describe()


# In[9]:


# 결측치 확인
df.isnull().sum() # 5 feature


# In[11]:


# 데이터 전처리 시작
# 명목형 변수 처리

df.describe(include='O') # 총 9개


# In[12]:


# engine 피쳐부터 시작

# value 종류부터 확인

df.engine.unique() # CC라는 단위가 존재


# In[13]:


# 메서드 str.split(expand)를 통해 컬럼 분리

df.engine.str.split(expand=True)


# In[14]:


# 컬럼 자체의 이름 설정

df[['engine','engine_unit']] = df.engine.str.split(expand=True)
df


# In[15]:


# engine을 실수형을 변형

df.engine = df.engine.astype('float32')
df.engine


# In[16]:


# (주의) 단위가 써있는 unit 행은 삭제(drop)

df.drop('engine_unit', axis = 1, inplace = True)
df


# In[17]:


df.columns


# In[18]:


# max_power(최대 출력) 처리 시작

df.max_power.head() # bhp 단위 수정


# In[19]:


df.max_power.unique() # 데이터 종류 확인


# In[20]:


# bhp 단일 value가 존재하여 전처리 필요


# In[21]:


df[['max_power','max_power_unit']] = df.max_power.str.split(expand=True)


# In[22]:


df


# In[23]:


# 직접 정의한 타입 변환 함수를 통해 error를 처리
# float로 변환
def handle_float(value):
    try:
        return float(value)
    except ValueError:
        return np.NaN


# In[24]:


df.max_power = df.max_power.apply(handle_float)


# In[25]:


df.max_power


# In[26]:


df.isnull().sum()


# In[27]:


# 단위는 불필요하니 삭제
# (주의)
df.drop('max_power_unit', axis = 1, inplace=True)


# In[28]:


df.head()


# In[29]:


# mileage 데이터 처리 시작

df.mileage.unique()


# In[30]:


# 배경 지식 필요 (kmpl)

df[['mileage','mileage_unit']]= df.mileage.str.split(expand=True)
df.head()


# In[31]:


df.mileage = df.mileage.astype('float32')
df.mileage


# In[32]:


df.fuel.unique() # fuel 4종류에 km/l과 km/kg가 대응함


# In[33]:


df.fuel[df.mileage_unit == 'kmpl'].unique() # kmpl : diesel, petrol 대응


# In[34]:


df.fuel[df.mileage_unit == 'km/kg'].unique() # km/kg에 lpg, cng가 대응


# In[37]:


# 현재 유가 확인

fuels = {
    'Petrol' : 80.4,
    'Diesel' : 73.5,
    'LPG' : 40.8,
    'CNG' : 44.2
}
fuels


# In[38]:


# map으로 딕셔너리 제작
df.fuel.map(fuels)


# In[39]:


# 연비
df.mileage / df.fuel.map(fuels)


# In[40]:


# apply로 행이 갖고 있는 열 사용

def handle_mileage(row):
    return row.mileage/fuels[row.fuel]


# In[42]:


df.mileage = df.apply(handle_mileage, axis=1)
df.mileage


# In[43]:


# unit 행 삭제 주의

df.drop('mileage_unit', axis=1, inplace= True)


# In[44]:


# torque 전처리
df.torque.unique()


# In[47]:


# unit을 추출하기 위한 함수 저으이

def handle_torque_unit(x):
    if 'NM' in str(x):
        return 'NM'
    if 'KGM' in str(x):
        return 'kgm'


# In[48]:


df.torque.apply(handle_torque_unit)


# In[49]:


# torque unit을 대문자 변환

df['torque_unit'] = df.torque.str.upper().apply(handle_torque_unit)


# In[53]:


# 100보다 큰 unit값 조회

df[df.torque_unit.isna()].torque.unique()


# In[55]:


df[df['torque_unit']=='Nm'].torque


# In[56]:


df[df['torque_unit']=='kgm'].torque


# In[57]:


# 결측치 강제처리
df.torque_unit.fillna('Nm', inplace=True)


# In[58]:


df.torque_unit.unique()


# In[59]:


# 정규표현식 사용

df.torque.str.extract("([\d\.]+)").astype('float') # 실수형으로 변환


# In[60]:


# 그외에는 torque에서 삭제

df.torque = df.torque.str.extract("([\d\.]+)").astype('float')
df.torque


# In[61]:


# 9.8066 kgm = 1Nm
# kgm을 Nm으로 만드는 함수 정의
def handle_torque_trans(x):
    if x.torque_unit =='kgm':
        return x.torque * 9.8066
    return x.torque


# In[62]:


# apply로 적용

df.torque = df.apply(handle_torque_trans, axis = 1)
df.torque


# In[63]:


# unit행은 불필요하므로 삭제 (주의)

df.drop('torque_unit', axis = 1, inplace = True)


# In[64]:


# name 수정

df.name.value_counts()


# In[65]:


df.name.unique()


# In[67]:


df.name = df.name.str.split(expand=True)[0]
df.name


# In[68]:


df.name.unique()


# In[69]:


# Land Rover 브랜드가 land만 조회됨
# replace로 교체
df.name = df.name.replace('Land', 'Land Rover')
df.name.unique()


# In[70]:


# 결측치 수정

df.isnull().sum()


# In[71]:


# 결측치는 제거

df.dropna(inplace=True)


# In[72]:


df.isnull().sum()


# In[74]:


# 더미 변수 생성

car_columns = df.describe(include=['O']).columns
car_columns


# In[75]:


# get_dummies

df = pd.get_dummies(df, columns = car_columns , drop_first= True)
df


# In[76]:


df # 더미 확인


# In[77]:


# 훈련셋/시험셋 분리 시작


# In[79]:


# 패키지 호출

from sklearn.model_selection import train_test_split


# In[80]:


X = df.drop('selling_price', axis = 1)
y = df['selling_price']


# In[81]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)


# In[84]:


# 모델 적용

from sklearn.ensemble import RandomForestRegressor


# In[85]:


# model 정의
model = RandomForestRegressor()
model.fit(X_train, y_train)


# In[86]:


# 훈련셋 예측

trainpred = model.predict(X_train)
testpred = model.predict(X_test)


# In[87]:


# 결정계수 호출

from sklearn.metrics import mean_squared_error


# In[88]:


mean_squared_error(y_train, trainpred, squared=False) # 훈련


# In[89]:


mean_squared_error(y_test, testpred, squared=False) # 시험

# overfitting 추론


# In[90]:


# KFold 교차검증 시작

from sklearn.model_selection import KFold


# In[91]:


kfold = KFold()


# In[93]:


# 인덱스 정리 시작

df.index # 빈 인덱스 정리필요


# In[95]:


df.reset_index()
df.reset_index(drop=True, inplace=True)


# In[96]:


df.index


# In[97]:


kf = KFold(n_splits = 5)
# kfold 교차검증 5회차 사용


# In[98]:


X = df.drop('selling_price', axis = 1)
y = df['selling_price'] # 변수들 재소환


# In[99]:


kf


# In[100]:


list(kf.split(X))


# In[102]:


# 함수 정의

for i, j in kf.split(X):
    print(f'i : {i}')
    print(f'j : {i}')


# In[103]:


# 순서대로 반복시키는 함수 정의

train_rmse_total = []
test_rmse_total= []


# In[105]:


# 반복문 정의
train_rmse_total = []
test_rmse_total= []


for train_index, test_index in kf.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = RandomForestRegressor(random_state=10)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = mean_squared_error(y_train, train_pred, squared=False) # 훈련셋 rmse 계산
    test_rmse = mean_squared_error(y_test, test_pred, squared= False) # 시험셋 rnmse 계산
    
    train_rmse_total.append(train_rmse)
    test_rmse_total.append(test_rmse)
    
    # 다소 시간 소요


# In[106]:


# 점수 종료

train_rmse_total, test_rmse_total


# In[107]:


sum(train_rmse_total) / len(train_rmse_total)# 평균


# In[108]:


sum(test_rmse_total) / len(test_rmse_total) # test 평균


# In[109]:


# 하이퍼 패러미터 조정

# 반복문 정의
train_rmse_total = []
test_rmse_total= []


for train_index, test_index in kf.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = RandomForestRegressor(
        n_estimators = 100,
        max_depth = 50,
        min_samples_split = 5,
        min_samples_leaf = 1,
        n_jobs = -1,
        random_state = 10)
    
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = mean_squared_error(y_train, train_pred, squared=False) # 훈련셋 rmse 계산
    test_rmse = mean_squared_error(y_test, test_pred, squared= False) # 시험셋 rnmse 계산
    
    train_rmse_total.append(train_rmse)
    test_rmse_total.append(test_rmse)
    
    # 다소 시간 소요


# In[116]:


# 실제값, 예측값 비교

"hyper_parameter train rmse",sum(train_rmse_total)/len(train_rmse_total)


# In[117]:


"hyper_parameter test rmse",sum(test_rmse_total)/len(test_rmse_total)


# In[118]:


# 하이퍼 패러미터로 예측 최적화 . regression 문제 종료.


# In[ ]:




