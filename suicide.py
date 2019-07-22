#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


    visualize the number of suicides of both male and females of different age group for every year from 1986-2016

    & find which age group has the highest number of suside rate.

    Which country has the highest number of Suside rate visualize it.

    Which country has highest & Lowest suside rate in 2006 visualize it.

    apply all classification algorithm and find best results,check the score and accuracy.then predict for this data

    country ----> Albania

    year ----> 2014

    Sex ----> Male

    Age ----> 24

    Suside no. ----> 22

    population ----> 279800

    country-year ----> 7.11

    gdp_for_year ($) ---->2,10,56,21,800

    gdp_per_capita ($) ---->749

    according to data find it was lie on which generation.

    Which country has the lowest GDP rate in 2014. visualize it.

Created on Mon Jul 22 08:34:59 2019

@author: vikram

country ----> Albania

   year ----> 2014

   Sex ----> Male

   Age ----> 24

   Suside no. ----> 22

   population ----> 279800

   country-year ----> 7.11

   gdp_for_year ($)  ---->2,10,56,21,800

   gdp_per_capita ($)  ---->749

   according to data find it was lie on which generation.

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('master.csv')
dataset.info()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('master.csv')
dataset.info()
def gdp_float(x):
  x=x.replace(",","")
  x=float(x)
  return (x)
                 
dataset[' gdp_for_year ($) ']=dataset[' gdp_for_year ($) '].apply(gdp_float)
#dataset.axes

# Encoding categorical data
features = dataset.iloc[:, :-1].values

labels = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features[:, 0] = labelencoder.fit_transform(features[:, 0])


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
features = onehotencoder.fit_transform(features).toarray()

df_3 = pd.get_dummies(features,drop_first=True)
features[0]




"""
dataset['country-year'].value_counts() #2321 differet coutry_year

dataset['country'].value_counts() #101 differet coutry

df_rank=dataset.groupby(['sex', 'age','suicides_no'])
df_rank.size()['age']
 """
#visualize the number of suicides of both male and females 
#of different age group for every year from 1986-2016

age_group = dataset['age'].unique().tolist()
suicides_no = []
for age_o in age_group:
    new_dataset = dataset[(dataset['sex']=='male') & (dataset['age']==age_o)].sort_values('suicides_no')
    suicides_no.append(new_dataset['suicides_no'].max())


#plt.style.use('fivethirtyeight')          
explode = (0, 0.3, 0, 0,0,0)  # explode 1st slice to 10% of the radius
plt.pie(suicides_no, explode=explode,labels=age_group, autopct='%.0f%%')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()



plt.xlabel("Ages")
plt.ylabel("SUicide_DATA")
plt.title("SUicide")
plt.barh(age_group,suicides_no,label='Suicides')
plt.legend()
plt.show()

# 2.Which country has the highest number of Suside rate visualize it.
age=dataset.groupby('age').aggregate({'suicides/100k pop': 'max'})
age.plot(kind='barh',figsize=[8,6],colormap='autumn')

#Which country has the highest number of Suside rate visualize it.
#cou=dataset.groupby('country').aggregate({'suicides/100k pop': 'max'})
#cou_max = dataset.sort_values('suicides/100k pop')


df_sorted= dataset.sort_values( by='suicides/100k pop', ascending = [False])
df_sorted['suicides/100k pop'][0:5]


plt.bar(df_sorted['country'][0:5],df_sorted['suicides/100k pop'][0:5], width = 0.5, align='center', alpha=1.0,)
plt.xlabel('Country Name')
plt.ylabel('SUiCIDE_rate')

plt.title('Country suicide_RATE')
 
plt.show()

# 3.Which country has highest & Lowest suside rate in 2006 visualize it.

cou_data = dataset[dataset['year']==2006].sort_values('suicides/100k pop')
list3 = cou_data['suicides/100k pop'].unique().tolist()
list4 = cou_data['country'].unique().tolist()
  
#mi suicide       
explode = (0.5,0,0,0,0,0,0,0,0)  # explode 1st slice to 10% of the radius
plt.pie(list3[0:9], explode=explode,labels=list4[0:9], autopct='%.0f%%')
plt.axis('equal')  
plt.show()

#max sucidie
explode = (0,0,0,0,0.3)  # explode 1st slice to 10% of the radius
plt.pie(list3[-5:], explode=explode,labels=list4[-5:], autopct='%.0f%%')
plt.axis('equal')  
plt.show()



#Which country has the lowest GDP rate in 2014. visualize it.

cou_gdp = dataset[(dataset['year']==2014)].sort_values(' gdp_for_year ($) ')
def gdp_float(x):
  x=x.replace(",","")
  x=float(x)
  return (x)
               
cou_gdp[' gdp_for_year ($) ']=cou_gdp[' gdp_for_year ($) '].apply(gdp_float)

explode = (0.9,0,0,0,0)  # explode 1st slice to 10% of the radius
plt.pie(cou_gdp[' gdp_for_year ($) '][0:60].unique(),explode=explode,labels=cou_gdp['country'][0:60].unique(), autopct='%.0f%%')
plt.axis('equal')  
plt.show()



"""
plt.xlabel("GDP %")
plt.ylabel("COUTRYA_GDP")
plt.title("GDP")
plt.bar(cou_gdp[' gdp_for_year ($) '][0:60].unique(),labels=cou_gdp['country'][0:60].unique(), width = 0.5, align='center', alpha=1.0,)
plt.legend()
plt.show()
"""
