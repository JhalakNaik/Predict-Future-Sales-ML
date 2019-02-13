# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import pickle

class feature_eng():
	
	def __init__(self):
		print("Preprocess object created")
	
	def _date_encoding(self,data):
		data['date']=pd.to_datetime(data['date'])
		data['year']=pd.DatetimeIndex(data['date']).year
		data['month']=pd.DatetimeIndex(data['date']).month
		data['day']=pd.DatetimeIndex(data['date']).day
		data['weekday'] = ((pd.DatetimeIndex(data['date']).dayofweek) // 5 == 1).astype(float)
		data.drop(['date'],axis=1,inplace=True)
		return data

	def _data_merge(self,data,item,cats,shops):
		data= pd.merge(data,shops,on='shop_id',how='left')
		data= pd.merge(data,item,on='item_id',how='left')
		data.drop(['item_name'],axis=1,inplace=True)
		data= pd.merge(data,cats,on='item_category_id',how='left')
	   	return data

	def _cat_type(self,cats):
		cats['split'] = cats['item_category_name'].str.split('-')
		cats['type'] = cats['split'].map(lambda x: x[0].strip())
		cats['type_code'] = LabelEncoder().fit_transform(cats['type'])
		cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
		cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
		cats = cats[['item_category_id','type_code', 'subtype_code']]
		return cats
	
	def _shop_city(self,shops):
		shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
		shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
		shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
		shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
		shops = shops[['shop_id','city_code']]
		return shops

	def _data_update(self,data):
		# Then, rename to match the proper cases
		# Якутск Орджоникидзе, 56
		data.loc[data['shop_id'] == 0, 'shop_id'] = 57
		# Якутск ТЦ "Центральный"
		data.loc[data['shop_id'] == 1, 'shop_id'] = 58
		# Жуковский ул. Чкалова 39м²
		data.loc[data['shop_id'] == 10, 'shop_id'] = 11
		return data

	def _type_data(self,shop_tr,data):
		shop_tr.drop(['City','Name'], axis=1, inplace=True)
		shop_tr['Type'] = LabelEncoder().fit_transform(shop_tr['Type'])
		data=pd.merge(data,shop_tr,on=['shop_id'],how='left')
		return data


	def calender_data(self,cal,data):
		cal['date']=pd.to_datetime(cal['date'])
		cal['year']=pd.DatetimeIndex(cal['date']).year
		cal['month']=pd.DatetimeIndex(cal['date']).month
		cal['day']=pd.DatetimeIndex(cal['date']).day
		cal.drop(['date'], axis=1, inplace=True)
		data = pd.merge(data, cal, on=['year','month','day'], how='left')
		data['holiday'] = data['holiday'].astype(np.int8)
		return data

class model():

	def __init__(self):
		print("Model train")

	def _model_train(self,test):
		filename = 'finalized_model.sav'
		loaded_model = pickle.load(open(filename, 'rb'))
		y_pred=loaded_model.predict(test)
		y_pred = [round(value) for value in y_pred]
		y_pred = np.clip(y_pred,0, 20)
		df=pd.DataFrame({'item_cnt_day':y_pred})
		test['item_cnt_day']=df.item_cnt_day
		t=test
		test=test.drop(['shop_id','date_block_num','item_id','item_price','month','day','item_category_id','year','weekday','city_code','type_code','subtype_code','holiday','Type'],axis=1,inplace=True)
		test=t
		test.to_csv('Final_lgbm.csv',index=False)
		ax = lgb.plot_importance(loaded_model, max_num_features=25)
		plt.show()

class ObjectOriented():

    def __init__(self,test, shops, items, item_cats,shop_tr, cal):
		self.test=test
		self.shops=shops
		self.items=items
		self.item_cats=item_cats
		self.shop_tr=shop_tr
		self.cal=cal
		self.feature_eng = feature_eng()
		self.model=model()
		self.test=self.feature_eng._date_encoding(self.test)
		self.item_cats=self.feature_eng._cat_type(self.item_cats)
		self.shops=self.feature_eng._shop_city(self.shops)
		self.test=self.feature_eng._data_update(self.test)
		self.test=self.feature_eng._data_merge(self.test,self.items,self.item_cats,self.shops)
		self.test=self.feature_eng._type_data(self.shop_tr,self.test)
		self.test=self.feature_eng.calender_data(self.cal,self.test)
		self.model._model_train(self.test)


if __name__ == "__main__":
	test = pd.read_csv("test.csv")
	shops =pd.read_csv("shops.csv")
	items= pd.read_csv("items.csv")
	item_cats=pd.read_csv("item_categories.csv")
	shop_tr= pd.read_csv("shops-translated.csv")
	cal=cal=pd.read_csv("calendar.csv")
	objectOriented=ObjectOriented(test, shops, items, item_cats, shop_tr, cal)
