# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

class eda():

	def __init__(self):
		print("EDA of data")

	def explore_data(self,data,test,shops,items,item_cats):
		print('training set: ', data.shape)
		print('test set: ', test.shape)
		print('num of shops: ', shops.shape)
		print('num of items: ',items.shape)
		print('num of item categories: ',item_cats.shape)

	
	def check_outlier(self,data1):
		plt.figure(figsize=(10,10))
		g = sns.boxplot( y=train['item_price'])
		g.set_yscale('log')
		plt.show()
		plt.figure(figsize=(10,10))
		g = sns.boxplot( y=train['item_cnt_day'])
		g.set_yscale('log')
		plt.show()
	
	def _eda_data(self,data):
	    print  data.isnull().sum()

	def _eda_corr(self,train):	
		corr_matrix = train.corr() # corr() : Pandas method, computes pairwise correlation of columns, excluding NA/null values
		print corr_matrix
		
	def _shop_month(self,train):
		grouped = pd.DataFrame(train.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index())
		fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))
		num_graph = 10
		id_per_graph = math.ceil(grouped.shop_id.max() / num_graph)
		count = 0
		for i in range(5):
			for j in range(2):
				sns.pointplot(x='date_block_num', y='item_cnt_day', hue='shop_id', data=grouped[np.logical_and(count*id_per_graph <= grouped['shop_id'], grouped['shop_id'] < (count+1)*id_per_graph)], ax=axes[i][j])
				count += 1
		plt.show()	

	def trends(self,train):
		plt.figure(figsize=(35,10))
		sns.countplot(x='shop_id', data=train)
		plt.title('Total Item Sale per Shop')	
		plt.show()

		sales_train_monthly = pd.DataFrame(train.groupby(['date_block_num'])['item_cnt_day'].sum().reset_index().sort_values('item_cnt_day'))
		plt.figure(figsize=(35,10))
		sns.barplot(x="date_block_num", y="item_cnt_day", data=sales_train_monthly , order=sales_train_monthly['date_block_num'])
		plt.title('Items sold per month')
		plt.show()
		
	def items_per_cat(self,item):
		# number of items per cat 
		x=item.groupby(['item_category_id']).count()
		x=x.sort_values(by='item_id',ascending=False)
		x=x.iloc[0:10].reset_index()
	
		# #plot
		plt.figure(figsize=(8,4))
		ax= sns.barplot(x.item_category_id, x.item_id, alpha=0.8)
		plt.title("Items per Category")
		plt.ylabel('# of items', fontsize=12)
		plt.xlabel('Category', fontsize=12)
		plt.show()

	def tot_sales_shop(self,train):
		ts=train.groupby(["date_block_num"])["item_cnt_day"].sum()
		ts.astype('float')
		plt.figure(figsize=(16,8))
		plt.title('Total Sales of the shop')
		plt.xlabel('Time')
		plt.ylabel('Sales')
		plt.plot(ts)
		plt.show()
	

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

	
	def _drop_duplicates(self,data):
		print('Before drop train shape:', data.shape)
		data.drop_duplicates(subset=['year','month','day','date_block_num', 'shop_id', 'item_id', 'item_cnt_day'], keep='first', inplace=True)
		data.reset_index(drop=True, inplace=True)
		print('After drop train shape:', data.shape)
		return data

	def _data_merge(self,data,item,cats,shops):
		data= pd.merge(data,shops,on='shop_id',how='left')
		data= pd.merge(data,item,on='item_id',how='left')
		data.drop(['item_name'],axis=1,inplace=True)
		data= pd.merge(data,cats,on='item_category_id',how='left')
	   	return data


	def _drop_outlier(self,data):
		data = data[(data['item_price'] > 0) & (data['item_price'] < 300000)]
		data= data[data.item_cnt_day<=1000]
		data.item_price[data.item_price<0]=1249.3
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

	def _type_train(self,shop_tr,train):
		shop_tr.drop(['City','Name'], axis=1, inplace=True)
		shop_tr['Type'] = LabelEncoder().fit_transform(shop_tr['Type'])
		train=pd.merge(train,shop_tr,on=['shop_id'],how='left')
		return train

	def _type_test(self,shop_tr,test):
		shop_tr['Type'] = LabelEncoder().fit_transform(shop_tr['Type'])
		test=pd.merge(test,shop_tr,on=['shop_id'],how='left')
		return test

	def calender_train(self,cal,train):
		cal['date']=pd.to_datetime(cal['date'])
		cal['year']=pd.DatetimeIndex(cal['date']).year
		cal['month']=pd.DatetimeIndex(cal['date']).month
		cal['day']=pd.DatetimeIndex(cal['date']).day
		cal.drop(['date'], axis=1, inplace=True)
		train = pd.merge(train, cal, on=['year','month','day'], how='left')
		train['holiday'] = train['holiday'].astype(np.int8)
		return train
	
	def calender_test(self,cal,test):
		test = pd.merge(test, cal, on=['year','month','day'], how='left')
		test['holiday'] = test['holiday'].astype(np.int8)
		return test


class model():

	def __init__(self):
		print("Model train")

	def _model_train(self,data,test):
		y=data['item_cnt_day']
		data.drop(['item_cnt_day'],axis=1,inplace=True)
		d_train = lgb.Dataset(data, label=y)
		params={}
		params['learning_rate'] = 0.03
		params['objective'] = 'regression'
		params['metric'] = 'rmse'
		params['num_leaves'] = 128
		params['min_data_in_leaf'] = 128
		params['seed']=1204
		params['verbose']=1
		params['colsample_bytree']= 0.75
		params['subsample']= 0.75
		params['bagging_seed']= 128
		params['bagging_freq']=1
		clf = lgb.train(params, d_train, 300,d_train,verbose_eval=10)
		y_pred=clf.predict(test)
		y_pred = [round(value) for value in y_pred]
		y_pred = np.clip(y_pred,0, 20)
		df=pd.DataFrame({'item_cnt_day':y_pred})
		test['item_cnt_day']=df.item_cnt_day
		t=test
		test=test.drop(['shop_id','date_block_num','item_id','item_price','month','day','item_category_id','year','weekday','city_code','type_code','subtype_code','holiday','Type'],axis=1,inplace=True)
		test=t
		test.to_csv('6dec_lgbm.csv',index=False)
		ax = lgb.plot_importance(clf, max_num_features=25)
		plt.show()
		import pickle
		filename = 'finalized_model.sav'
		pickle.dump(clf, open(filename, 'wb'))

class ObjectOriented():

    def __init__(self, train, test, shops, items, item_cats,shop_tr, cal):
		self.train=train
		self.test=test
		self.shops=shops
		self.items=items
		self.item_cats=item_cats
		self.shop_tr=shop_tr
		self.cal=cal
		self.eda_obj=eda()
		self.feature_eng = feature_eng()
		self.model=model()
		self.eda_obj.explore_data(self.train,self.test,self.shops,self.items,self.item_cats)
		self.eda_obj._eda_data(self.train)	
		self.eda_obj.check_outlier(self.train)
		self.eda_obj._eda_data(self.train)
		self.eda_obj._eda_corr(self.train)
		self.eda_obj.trends(self.train)
		self.eda_obj._shop_month(self.train)
		self.eda_obj.items_per_cat(self.items)
		self.eda_obj.tot_sales_shop(self.train)
		self.train=self.feature_eng._date_encoding(self.train)
		self.test=self.feature_eng._date_encoding(self.test)
		self.train=self.feature_eng._drop_duplicates(self.train)
		self.train=self.feature_eng._drop_outlier(self.train)
		self.item_cats=self.feature_eng._cat_type(self.item_cats)
		self.shops=self.feature_eng._shop_city(self.shops)
		self.train=self.feature_eng._data_update(self.train)
		self.test=self.feature_eng._data_update(self.test)
		self.train=self.feature_eng._data_merge(self.train,self.items,self.item_cats,self.shops)
		self.test=self.feature_eng._data_merge(self.test,self.items,self.item_cats,self.shops)
		self.train=self.feature_eng._type_train(self.shop_tr,self.train)
		self.test=self.feature_eng._type_test(self.shop_tr,self.test)
		self.train=self.feature_eng.calender_train(self.cal,self.train)
		self.test=self.feature_eng.calender_test(self.cal,self.test)
		self.model._model_train(self.train,self.test)


if __name__ == "__main__":
	train = pd.read_csv("train.csv")
	test = pd.read_csv("test.csv")
	shops =pd.read_csv("shops.csv")
	items= pd.read_csv("items.csv")
	item_cats=pd.read_csv("item_categories.csv")
	shop_tr= pd.read_csv("shops-translated.csv")
	cal=cal=pd.read_csv("calendar.csv")
	objectOriented=ObjectOriented(train, test, shops, items, item_cats, shop_tr, cal)
