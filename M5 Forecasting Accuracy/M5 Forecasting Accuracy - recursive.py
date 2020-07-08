# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:20:41 2020

@author: dylan
"""

import pandas as pd
import time as t

t1 = t.time()

path = "C:/Users/dylan/Datasets/Kaggle/m5-forecasting-accuracy"

# Prices at store/product id/week level
prices = pd.read_csv(path+'/sell_prices.csv')

# Events on the date level
cal = pd.read_csv(path+'/calendar.csv')

# Units sold of each product, each day has its own column
# Columns that aren't dates are put in the index for later transformation
stv = pd.read_csv(path+'/sales_train_validation.csv')
stv_index = stv.columns[0:6].tolist()
stv.set_index(stv_index, inplace=True)

# Transpose sales so that date is all in one column
stv2 = stv.stack().reset_index()
stv2 = stv2.rename(columns = {'level_6':'d', 0:'sales'})
#stv_sample = stv2.sample(frac=.01)

# Merge with prices and calendar
fdata = stv2.merge(cal, how='left')
fdata = fdata.merge(prices, how='left')
fdata.sort_values(['id','d'], inplace=True)

# Create extra vars
# (Sales/price lags and avgs, event dummies, seasonality, geography)
# Be sure to group lags by id
# Prices only change weekly, not daily
            # Lag sales
for i in range(7):
    fdata['sales_lag{0}'.format(i+1)] = fdata.groupby('id')['sales'].shift(i+1)
            # Rolling average sales
fdata['avg_sales_1week'] = fdata.groupby('id')['sales'].rolling(7).mean()\
                            .reset_index(level=0, drop=True)
fdata['avg_sales_1month'] = fdata.groupby('id')['sales'].rolling(30).mean()\
                            .reset_index(level=0, drop=True)
fdata['avg_sales_1quarter'] = fdata.groupby('id')['sales'].rolling(90).mean()\
                            .reset_index(level=0, drop=True)
            # Lag price
fdata['price_lag_week'] = fdata.groupby('id')['sell_price'].shift(7)
fdata['price_lag_month'] = fdata.groupby('id')['sell_price'].shift(30)
            # Rolling average price
fdata['avg_price_month'] = fdata.groupby('id')['sell_price'].rolling(30).mean()\
                            .reset_index(level=0, drop=True)
fdata['avg_price_quarter'] = fdata.groupby('id')['sell_price'].rolling(90).mean()\
                            .reset_index(level=0, drop=True)
            # Create dummies
dept_dummies = pd.get_dummies(fdata['dept_id'])
category_dummies = pd.get_dummies(fdata['cat_id'])
store_dummies = pd.get_dummies(fdata['store_id'])
state_dummies = pd.get_dummies(fdata['state_id'])
dow_dummies = pd.get_dummies(fdata['weekday'])
month_dummies = pd.get_dummies(fdata['month'], prefix='month')
concat_list = [fdata, dept_dummies, category_dummies, store_dummies,\
               state_dummies, dow_dummies, month_dummies]
fdata = pd.concat(concat_list, axis='columns')
            # Create event dummies
eventname1_dummies = pd.get_dummies(fdata['event_name_1'])
eventname2_dummies = pd.get_dummies(fdata['event_name_2'])
eventtype1_dummies = pd.get_dummies(fdata['event_type_1'])
eventtype2_dummies = pd.get_dummies(fdata['event_type_2'])
events = set()
events.update(cal.event_name_1.dropna().unique().tolist())
events.update(cal.event_name_2.dropna().unique().tolist())
for event in events:
    try:
        fdata[event] = eventname1_dummies[event] + eventname2_dummies[event]
    except:
        try:
            fdata[event] = eventname1_dummies[event]
        except:
            fdata[event] = eventname2_dummies[event]
eventtypes = set()
eventtypes.update(cal.event_type_1.dropna().unique().tolist())
eventtypes.update(cal.event_type_2.dropna().unique().tolist())
for eventtype in eventtypes:
    try:
        fdata[eventtype] = eventtype1_dummies[eventtype] + eventtype2_dummies[eventtype]
    except:
        try:
            fdata[eventtype] = eventtype1_dummies[eventtype]
        except:
            fdata[eventtype] = eventtype2_dummies[eventtype]

t2 = t.time()

runtime = (t2-t1)/60


fdata.to_hdf(path+'/full_data_clean.h5', 'fdata')
# AnAlysIS

# Set predictors and target (price and sales)

# Train models (one for price, one for sales)

# Forecast (repeat these steps for each day of forecast)
    # Create a new observations for each id
    # Predict price for each new observation
    # Create lag vars
    # Add day/event/state/cat/dept/store dummies
    # Predict sales
    










