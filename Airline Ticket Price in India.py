#!/usr/bin/env python
# coding: utf-8

# # Airline Ticket Price 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import sys


# In[3]:


bom_blr_data = pd.read_csv(r"C:\Users\jki\Downloads\flight_data_BOM_BLR.csv")
bom_blr_data


# In[4]:


bom_blr_data.shape


# In[5]:


bom_blr_data.info()


# In[6]:


bom_blr_data.describe()


# In[7]:


del_blr_data = pd.read_csv(r"C:\Users\jki\Downloads\flight_data_DEL_BLR.csv")
del_blr_data


# In[8]:


del_blr_data.shape


# In[9]:


del_blr_data.info()


# In[10]:


del_blr_data.describe()


# In[11]:


del_bom_data = pd.read_csv(r"C:\Users\jki\Downloads\flight_data_DEL_BOM.csv")
del_bom_data


# In[12]:


del_bom_data.shape


# In[14]:


del_bom_data.info()


# In[15]:


del_bom_data.describe()


# In[16]:


del_ccu_data = pd.read_csv(r"C:\Users\jki\Downloads\flight_data_DEL_CCU.csv")
del_ccu_data


# In[17]:


del_ccu_data.shape


# In[18]:


del_ccu_data.info()


# In[19]:


del_ccu_data.describe()


# In[20]:


del_hyd_data = pd.read_csv(r"C:\Users\jki\Downloads\flight_data_DEL_HYD.csv")
del_hyd_data


# In[21]:


del_hyd_data.shape


# In[22]:


del_hyd_data.info()


# In[23]:


del_hyd_data.describe()


# In[24]:


bom_blr_data["Price"] = bom_blr_data["Price"].str.replace(",", "")
del_blr_data["Price"] = del_blr_data["Price"].str.replace(",", "")
del_bom_data["Price"] = del_bom_data["Price"].str.replace(",", "")
del_ccu_data["Price"] = del_ccu_data["Price"].str.replace(",", "")
del_hyd_data["Price"] = del_hyd_data["Price"].str.replace(",", "")


# In[25]:


bom_blr_data = bom_blr_data.astype({"Price":int})
del_blr_data = del_blr_data.astype({"Price":int})
del_bom_data = del_bom_data.astype({"Price":int})
del_ccu_data = del_ccu_data.astype({"Price":int})
del_hyd_data = del_hyd_data.astype({"Price":int})


# In[26]:


bom_blr_data["DepartingHour"] = bom_blr_data["DepartingTime"].str.split(":").str[0]
del_blr_data["DepartingHour"] = del_blr_data["DepartingTime"].str.split(":").str[0]
del_bom_data["DepartingHour"] = del_bom_data["DepartingTime"].str.split(":").str[0]
del_ccu_data["DepartingHour"] = del_ccu_data["DepartingTime"].str.split(":").str[0]
del_hyd_data["DepartingHour"] = del_hyd_data["DepartingTime"].str.split(":").str[0]


# In[27]:


bom_blr_data = bom_blr_data.astype({"DepartingHour" : int})
del_blr_data = del_blr_data.astype({"DepartingHour" : int})
del_bom_data = del_bom_data.astype({"DepartingHour" : int})
del_ccu_data = del_ccu_data.astype({"DepartingHour" : int})
del_hyd_data = del_hyd_data.astype({"DepartingHour" : int})


# # Pairplot method

# In[28]:


sns.pairplot(bom_blr_data, kind='scatter', height=3.5)


# In[29]:


sns.pairplot(del_blr_data, kind='scatter', height=3.5)


# In[30]:


sns.pairplot(del_bom_data, kind='scatter', height=3.5)


# In[31]:


sns.pairplot(del_ccu_data, kind='scatter', height=3.5)


# In[33]:


sns.pairplot(del_hyd_data, kind='scatter', height=3.5)


# # lmplot method between DepartingHour vs Price
# 

# In[34]:


sns.lmplot(bom_blr_data, x='DepartingHour', y='Price', line_kws = {'color':'black'}, scatter_kws = {'color':'darkgrey'})
plt.title('DepartingHour vs Price')


# In[35]:


sns.lmplot(del_blr_data, x='DepartingHour', y='Price', line_kws = {'color':'black'}, scatter_kws = {'color':'darkgrey'})
plt.title('DepartingHour vs Price')


# In[36]:


sns.lmplot(del_bom_data, x='DepartingHour', y='Price', line_kws = {'color':'black'}, scatter_kws = {'color':'darkgrey'})
plt.title('DepartingHour vs Price')


# In[37]:


sns.lmplot(del_ccu_data, x='DepartingHour', y='Price', line_kws = {'color':'black'}, scatter_kws = {'color':'darkgrey'})
plt.title('DepartingHour vs Price')


# In[38]:


sns.lmplot(del_hyd_data, x='DepartingHour', y='Price', line_kws = {'color':'black'}, scatter_kws = {'color':'darkgrey'})
plt.title('DepartingHour vs Price')


# # Barplot method between FlightName vs Price
# 

# In[39]:


bom_blr_data_avg_price = bom_blr_data.groupby('FlightName')['Price'].mean().reset_index().sort_values(by='Price', ascending=False)
sns.barplot(x='FlightName', y='Price', data=bom_blr_data_avg_price)
plt.title('Barplot FlightName vs Avg. Price')


# In[40]:


del_blr_data_avg_price = del_blr_data.groupby('FlightName')['Price'].mean().reset_index().sort_values(by='Price', ascending=False)
sns.barplot(x='FlightName', y='Price', data=del_blr_data_avg_price)
plt.title('Barplot FlightName vs Avg. Price')


# In[41]:


del_bom_data_avg_price = del_bom_data.groupby('FlightName')['Price'].mean().reset_index().sort_values(by='Price', ascending=False)
sns.barplot(x='FlightName', y='Price', data=del_bom_data_avg_price)
plt.title('Barplot FlightName vs Avg. Price')


# In[42]:


del_ccu_data_avg_price = del_ccu_data.groupby('FlightName')['Price'].mean().reset_index().sort_values(by='Price', ascending=False)
sns.barplot(x='FlightName', y='Price', data=del_ccu_data_avg_price)
plt.title('Barplot FlightName vs Avg. Price')


# In[43]:


del_hyd_data_avg_price = del_hyd_data.groupby('FlightName')['Price'].mean().reset_index().sort_values(by='Price', ascending=False)
sns.barplot(x='FlightName', y='Price', data=del_hyd_data_avg_price)
plt.title('Barplot FlightName vs Avg. Price')


# # Pie Chart method using for FlightName

# In[44]:


bom_blr_data.groupby("FlightName").size().plot(kind='pie', autopct='%1.0f%%', subplots=True)
plt.title('Pie chart Bombay --> Bengaluru Flight')
plt.show()


# In[45]:


del_blr_data.groupby("FlightName").size().plot(kind='pie', autopct='%1.0f%%', subplots=True)
plt.title('Pie chart Delhi --> Bengaluru Flight')
plt.show()


# In[46]:


del_bom_data.groupby("FlightName").size().plot(kind='pie', autopct='%1.0f%%', subplots=True)
plt.title('Pie chart Delhi --> Bombay Flight')
plt.show()


# In[47]:


del_ccu_data.groupby("FlightName").size().plot(kind='pie', autopct='%1.0f%%', subplots=True)
plt.title('Pie chart Delhi --> Kolkata Flight')
plt.show()


# In[48]:


del_hyd_data.groupby("FlightName").size().plot(kind='pie', autopct='%1.0f%%', subplots=True)
plt.title('Pie chart Delhi --> Hydrabad Flight')
plt.show()


# # Lineplot method using DepartingHour, Price and ArrivingCity
# 

# In[49]:


sns.lineplot(bom_blr_data, x='DepartingHour', y='Price', hue='ArrivingCity', marker='o')


# In[50]:


sns.lineplot(del_blr_data, x='DepartingHour', y='Price', hue='ArrivingCity', marker='o')


# In[51]:


sns.lineplot(del_bom_data, x='DepartingHour', y='Price', hue='ArrivingCity', marker='o')


# In[52]:


sns.lineplot(del_ccu_data, x='DepartingHour', y='Price', hue='ArrivingCity', marker='o')


# In[53]:


sns.lineplot(del_hyd_data, x='DepartingHour', y='Price', hue='ArrivingCity', marker='o')


# # Polynomial Regression
# 

# In[54]:


bom_blr_x = bom_blr_data['DepartingHour']
bom_blr_y = bom_blr_data['Price']

model = np.poly1d(np.polyfit(bom_blr_x, bom_blr_y, 3))
line = np.linspace(1, 30, 90)


# In[55]:


plt.scatter(bom_blr_x, bom_blr_y)
plt.plot(line, model(line))
plt.title('Polynomial Regression [Bombay --> Bengaluru]')
plt.show()


# In[56]:


del_blr_x = del_blr_data['DepartingHour']
del_blr_y = del_blr_data['Price']

model = np.poly1d(np.polyfit(del_blr_x, del_blr_y, 3))
line = np.linspace(1, 30, 95)


# In[57]:


plt.scatter(del_blr_x, del_blr_y)
plt.plot(line, model(line))
plt.title('Polynomial Regression [Delhi --> Bengaluru]')
plt.show()


# In[58]:


del_bom_x = del_bom_data['DepartingHour']
del_bom_y = del_bom_data['Price']

model = np.poly1d(np.polyfit(del_bom_x, del_bom_y, 3))
line = np.linspace(1, 30, 100)


# In[59]:


plt.scatter(del_bom_x, del_bom_y)
plt.plot(line, model(line))
plt.title('Polynomial Regression [Delhi --> Bombay]')
plt.show()


# In[60]:


del_ccu_x = del_ccu_data['DepartingHour']
del_ccu_y = del_ccu_data['Price']

model = np.poly1d(np.polyfit(del_ccu_x, del_ccu_y, 3))
line = np.linspace(1, 30, 60)


# In[61]:


plt.scatter(del_ccu_x, del_ccu_y)
plt.plot(line, model(line))
plt.title('Polynomial Regression [Delhi --> Kolkata]')
plt.show()


# In[62]:


del_hyd_x = del_hyd_data['DepartingHour']
del_hyd_y = del_hyd_data['Price']

model = np.poly1d(np.polyfit(del_hyd_x, del_hyd_y, 3))
line = np.linspace(1, 30, 59)


# In[63]:


plt.scatter(del_hyd_x, del_hyd_y)
plt.plot(line, model(line))
plt.title('Polynomial Regression [Delhi --> Hydrabad]')
plt.show()


# # Decision Tree method

# In[64]:


bom_blr_d = {'Air India':0, 'AirAsia':1, 'IndiGo':2, 'Akasa Air':3, 'Vistara':4}
bom_blr_data['FlightName'] = bom_blr_data['FlightName'].map(bom_blr_d)


# In[65]:


bom_blr_d = {'Bengaluru':0, 'Mumbai':1}
bom_blr_data['ArrivingCity'] = bom_blr_data['ArrivingCity'].map(bom_blr_d)


# In[66]:


features = ['DepartingHour', 'Price', 'FlightName', 'ArrivingCity']
bom_blr_x = bom_blr_data[features]
bom_blr_y = bom_blr_data['ArrivingCity']


# In[67]:


bom_blr_dtree = DecisionTreeClassifier()
bom_blr_dtree = bom_blr_dtree.fit(bom_blr_x, bom_blr_y)


# In[68]:


tree.plot_tree(bom_blr_dtree, feature_names=features)


# In[69]:


del_blr_d = {'Akasa Air':0, 'AirAsia':1, 'IndiGo':2, 'Air India':3, 'SpiceJet':4, 'Vistara':5}
del_blr_data['FlightName'] = del_blr_data['FlightName'].map(del_blr_d)


# In[70]:


del_blr_d = {'Bengaluru':0, 'New Delhi':1}
del_blr_data['ArrivingCity'] = del_blr_data['ArrivingCity'].map(del_blr_d)


# In[71]:


features = ['DepartingHour', 'Price', 'FlightName', 'ArrivingCity']
del_blr_x = del_blr_data[features]
del_blr_y = del_blr_data['ArrivingCity']


# In[72]:


del_blr_dtree = DecisionTreeClassifier()
del_blr_dtree = del_blr_dtree.fit(del_blr_x, del_blr_y)


# In[73]:


tree.plot_tree(del_blr_dtree, feature_names=features)


# In[74]:


del_bom_d = {'Akasa Air':0, 'IndiGo':1, 'Air India':2, 'SpiceJet':3, 'Vistara':4}
del_bom_data['FlightName'] = del_bom_data['FlightName'].map(del_bom_d)


# In[75]:


del_bom_d = {'Mumbai':0, 'New Delhi':1}
del_bom_data['ArrivingCity'] = del_bom_data['ArrivingCity'].map(del_bom_d)


# In[76]:


features = ['DepartingHour', 'Price', 'FlightName', 'ArrivingCity']
del_bom_x = del_bom_data[features]
del_bom_y = del_bom_data['ArrivingCity']


# In[77]:


del_bom_dtree = DecisionTreeClassifier()
del_bom_dtree = del_bom_dtree.fit(del_bom_x, del_bom_y)


# In[78]:


tree.plot_tree(del_bom_dtree, feature_names=features)


# In[79]:


del_ccu_d = {'IndiGo':0, 'Vistara':1, 'SpiceJet':2, 'Air India':3}
del_ccu_data['FlightName'] = del_ccu_data['FlightName'].map(del_ccu_d)


# In[80]:


del_ccu_d = {'Kolkata':0, 'New Delhi':1}
del_ccu_data['ArrivingCity'] = del_ccu_data['ArrivingCity'].map(del_ccu_d)


# In[81]:


features = ['DepartingHour', 'Price', 'FlightName', 'ArrivingCity']
del_ccu_x = del_ccu_data[features]
del_ccu_y = del_ccu_data['ArrivingCity']


# In[82]:


del_ccu_dtree = DecisionTreeClassifier()
del_ccu_dtree = del_ccu_dtree.fit(del_ccu_x, del_ccu_y)




# In[83]:


tree.plot_tree(del_ccu_dtree, feature_names=features)


# In[84]:


del_hyd_d = {'Akasa Air':0, 'IndiGo':1, 'Air India':2, 'Vistara':3, 'SpiceJet':4}
del_hyd_data['FlightName'] = del_hyd_data['FlightName'].map(del_hyd_d)


# In[85]:


del_hyd_d = {'Hyderabad':0, 'New Delhi':1}
del_hyd_data['ArrivingCity'] = del_hyd_data['ArrivingCity'].map(del_hyd_d)


# In[86]:


features = ['DepartingHour', 'Price', 'FlightName', 'ArrivingCity']
del_hyd_x = del_hyd_data[features]
del_hyd_y = del_hyd_data['ArrivingCity']


# In[87]:


del_hyd_dtree = DecisionTreeClassifier()
del_hyd_dtree = del_hyd_dtree.fit(del_hyd_x, del_hyd_y)


# In[88]:


tree.plot_tree(del_hyd_dtree, feature_names=features)


# # Catplot Method using DepartingHour, Price and DepartingCity¶
# 

# In[89]:


sns.catplot(bom_blr_data, x='DepartingHour', y='Price', hue='DepartingCity', height=5.5)


# In[90]:


sns.catplot(del_blr_data, x='DepartingHour', y='Price', hue='DepartingCity', height=5.5)


# In[91]:


sns.catplot(del_bom_data, x='DepartingHour', y='Price', hue='DepartingCity', height=5.5)


# In[92]:


sns.catplot(del_ccu_data, x='DepartingHour', y='Price', hue='DepartingCity', height=5.5)


# In[93]:


sns.catplot(del_hyd_data, x='DepartingHour', y='Price', hue='DepartingCity', height=5.5)


# # Violinplot method using ArrivingCity, Price and DepartingCity¶
# 

# In[94]:


sns.violinplot(bom_blr_data, x='ArrivingCity', y='Price', hue='DepartingCity', dodge=False)


# In[95]:


sns.violinplot(del_blr_data, x='ArrivingCity', y='Price', hue='DepartingCity', dodge=False)


# In[96]:


sns.violinplot(del_bom_data, x='ArrivingCity', y='Price', hue='DepartingCity', dodge=False)


# In[97]:


sns.violinplot(del_ccu_data, x='ArrivingCity', y='Price', hue='DepartingCity', dodge=False)


# In[98]:


sns.violinplot(del_hyd_data, x='ArrivingCity', y='Price', hue='DepartingCity', dodge=False)


# In[ ]:




