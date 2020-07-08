# -*- coding: utf-8 -*-
"""
Created on Fri May 15 21:59:33 2020

@author: dylan
"""

import pandas as pd
import time as t
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

t1 = t.time()

path = "C:/Users/dylan/Datasets/Kaggle/m5-forecasting-accuracy"

# Prices at store/product id/week level
prices = pd.read_csv(path+'/sell_prices.csv')

# Events on the date level
cal = pd.read_csv(path+'/calendar.csv')

# Units sold of each product, each day has its own column
stv = pd.read_csv(path+'/sales_train_validation.csv')

# For each store
stv.store_id.unique()
store = 'CA_1'
# Get stv store data
sdata = stv[stv['store_id']==store]
# Transpose sdata
sdata_index = sdata.columns[0:6].tolist()
sdata.set_index(sdata_index, inplace=True)
sdata = sdata.stack().reset_index()
sdata = sdata.rename(columns = {'level_6':'d', 0:'sales'})
# Merge datasets into one
sdata = sdata.merge(cal, how='left')
sdata = sdata.merge(prices, how='left')
# Create variables (NO LAGS)
            # Create dummies
dept_dummies = pd.get_dummies(sdata['dept_id'])
category_dummies = pd.get_dummies(sdata['cat_id'])
dow_dummies = pd.get_dummies(sdata['weekday'])
month_dummies = pd.get_dummies(sdata['month'], prefix='month')
concat_list = [sdata, dept_dummies, category_dummies,\
               dow_dummies, month_dummies]
sdata = pd.concat(concat_list, axis='columns')
            # Create event dummies
eventname1_dummies = pd.get_dummies(sdata['event_name_1'])
eventname2_dummies = pd.get_dummies(sdata['event_name_2'])
eventtype1_dummies = pd.get_dummies(sdata['event_type_1'])
eventtype2_dummies = pd.get_dummies(sdata['event_type_2'])
events = set()
events.update(cal.event_name_1.dropna().unique().tolist())
events.update(cal.event_name_2.dropna().unique().tolist())
for event in events:
    try:
        sdata[event] = eventname1_dummies[event] + eventname2_dummies[event]
    except:
        try:
            sdata[event] = eventname1_dummies[event]
        except:
            sdata[event] = eventname2_dummies[event]
eventtypes = set()
eventtypes.update(cal.event_type_1.dropna().unique().tolist())
eventtypes.update(cal.event_type_2.dropna().unique().tolist())
for eventtype in eventtypes:
    try:
        sdata[eventtype] = eventtype1_dummies[eventtype] + eventtype2_dummies[eventtype]
    except:
        try:
            sdata[eventtype] = eventtype1_dummies[eventtype]
        except:
            sdata[eventtype] = eventtype2_dummies[eventtype]

sdata['d_num'] = sdata['d'].str.split(pat='_',expand=True)[1]
nulls = sdata.isnull().sum().sort_values(ascending=False)

# Analysis (without price)
y = sdata['sales']
non_features = ['id','item_id','dept_id','cat_id','store_id','state_id'
                ,'date','wm_yr_wk','weekday','wday','month','event_name_1'
                ,'event_type_1','event_name_2','event_type_2','d','sell_price']
reference_dummies = ['FOODS_1','FOODS','Monday','month_1']
non_features.extend(reference_dummies)
X = sdata.drop(columns=non_features)

x_train, y_train, x_test, y_test =  train_test_split(X,y,test_size=0.25,stratify=)

model = LinearRegression()
model.fit(X,y)
model.score(X,y)



















