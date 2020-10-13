#!/usr/bin/env python
# coding: utf-8

# # 2차데이터에대한 이동평균값


import pandas as pd 


df = pd.read_csv('dataset_1.csv' , encoding='CP949')

# df.head(5)

df_Quantity = df[['코드' , '비식별' , '실사용량' , '년']]

grouped_Quantity = df_Quantity.groupby(['약품코드' , '비식별_병원' , '년' ])

Quantity_result = grouped_Quantity.sum().unstack(level = -1).fillna(0).reset_index()

# Quantity_result.head(5)

Quantity_result.columns = Quantity_result.columns.get_level_values(1)

cols = Quantity_result.columns.tolist()
cols[0] = '코드'
cols[1] = '비식별'

Quantity_result.columns = cols
Quantity_result


Quantity_result['sum'] = Quantity_result.iloc[:,:].sum(axis=1)
Quantity_result = Quantity_result[Quantity_result['sum']!= 0.0]
Quantity_result.drop('sum' ,axis=1 , inplace=True)


# Quantity_result.head(5)



Quantity_result.to_csv('...' , encoding = 'CP949' , index =False)

Quantity_result_T = Quantity_result.T


# Quantity_result_T


Quantity_result_T.columns = [Quantity_result_T.iloc[0] , Quantity_result_T.iloc[1]]

Quantity_result_T = Quantity_result_T.iloc[2:]

Quantity_result_T

Quantity_result_T_mvAVG=Quantity_result_T.rolling(5).mean().T.reset_index()

# Quantity_result_T_mvAVG


Quantity_result_T_mvAVG.to_csv('...' , encoding = 'CP949' , index =False)

##

df_Count = df[['코드' , '비식별' , '실건수' , '년']]

grouped_Count = df_Count.groupby(['코드' , '비식별' , '년' ])

Count_result = grouped_Count.sum().unstack(level = -1).fillna(0).reset_index()

# Count_result.head(5)

Count_result.columns = Count_result.columns.get_level_values(1)

cols = Count_result.columns.tolist()
cols[0] = '코드'
cols[1] = '비식별'

Count_result.columns = cols
# Count_result


Count_result['sum'] = Count_result.iloc[:,:].sum(axis=1)

Count_result = Count_result[Count_result['sum']!= 0.0]
Count_result.drop('sum' ,axis=1 , inplace=True)

# Count_result.head(5)


Count_result.to_csv('...' , encoding = 'CP949' , index =False)

Count_result_T = Count_result.T

# Count_result_T.head(5)

Count_result_T.columns = [Count_result_T.iloc[0] , Count_result_T.iloc[1]]

# Count_result_T

Count_result_T = Count_result_T.iloc[2:]


Count_result_T_mvAVG=Count_result_T.rolling(5).mean().T.reset_index()


Count_result_T_mvAVG.to_csv('...' , encoding = 'CP949' , index =False)


# BEFORE PREPROCESSING
# 년 , 월 , 코드 . 기타 양에 대한 값 , 비식별 , 기타값
# 2012	5	C1	mL	4	240.0	0	0.0	0	0.0	0	0.0	C011	240.0	4
# 2012	5	C1	mL	1	24.0	0	0.0	0	0.0	0	0.0	C012	24.0	1
# 2012	5	C2	tab	0	0.0	0	0.0	0	0.0	0	0.0	C013	0.0	0
# 2012	5	C2	tab	1	3.0	0	0.0	0	0.0	0	0.0	C014	3.0	1
# 2012	5	C2	tab	2	6.0	0	0.0	1	3.0	0	0.0	C015	3.0	1
# ...

#

# AFTER PREPROCESSING
# 코드	비식별	2012	2013	2014	2015	2016	2017	2018	2019 (기타값들의 합 )
# 	C1	C011	0.0	0.0	0.0	0.0	0.0	5.0	0.0	0.0
# 	C1	C012	0.0	0.0	0.0	0.0	0.0	833.0	1.0	0.0
# 	C2	C013	0.0	0.0	0.0	0.0	0.0	4.0	0.0	0.0
# 	C2	C014	0.0	0.0	0.0	0.0	0.0	22.0	0.0	0.0
# 	C2	C015	0.0	0.0	0.0	0.0	0.0	0.0	0.0	2614.0
#


