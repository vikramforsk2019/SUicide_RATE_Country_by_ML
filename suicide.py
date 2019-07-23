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

    country-year ----> 7.11 #remoove it

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
features = dataset.iloc[:, [0,1,2,3,4,5,6,9,10]].values

labels = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
#for coutry
labelencoder1 = LabelEncoder()
features[:, 0] = labelencoder1.fit_transform(features[:, 0])

# for  sex
labelencoder2 = LabelEncoder()
features[:, 2] = labelencoder2.fit_transform(features[:, 2])

#for age group
labelencoder3 = LabelEncoder()
features[:, 3] = labelencoder2.fit_transform(features[:, 3])



from sklearn.preprocessing import OneHotEncoder
#for coutry
cou_hot = OneHotEncoder(categorical_features = [0])
features = cou_hot.fit_transform(features).toarray()
# dropping first column
features = features[:, 1:]

#for age group
age_hot = OneHotEncoder(categorical_features = [102])
features = age_hot.fit_transform(features).toarray()
features = features[:, 1:]

#1.Albania
le = labelencoder1.transform(['Albania'])
ohe = cou_hot.transform(le.reshape(1,1)).toarray()
ohe[0][1:]

#2.year
year = 2014
#Sex ----> Male
sex = 1
#Age ----> 24
le2 = np.array([0,0,0,0,0]) 
#laelecode do not  work  give alwayes 0 elemet we direct give the label no
#age2 = age_hot.transform(le2.reshape(1,1)).toarray()
#age[0][1:]
#3.Suside no. ----> 22
Suside_no = 22
#4.population ----> 279800
population = 279800
 
gdp_rate = 4.5
#5 gdp_for_year ($)  ---->2,10,56,21,800
gdp_year = 2105621800
#6.gdp_per_capita ($)  ---->749
gdp_cap = 749
 
[print(i) for i in le2]

for i in ohe:
    list(i)

x = [(i for i in le2, int),year,sex,Suside_no,population,gdp_rate,gdp_year,gdp_cap]
 
a = np.fromiter( [ x for x in range(0,4) ], int )
 = 22
dept_name=input('enter the coutry name>')
work_hour=input('enter the hours>')

le = labelencoder.transform([dept_name])
ohe = onehotencoder.transform(le.reshape(1,1)).toarray()
x = [ohe[0][1],ohe[0][2],int(work_hour),1,3]
x = np.array(x)


105-year
104-coutry ed
106-sex
107-sucide o







from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(features_train, labels_train)


#Calculate Class Probabilities
probability = classifier.predict_proba(features_test)

# Predicting the class labels
labels_pred = classifier.predict(features_test)

pd.DataFrame(labels_pred, labels_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(accuracy_score(labels_test, labels_pred)*100)



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
