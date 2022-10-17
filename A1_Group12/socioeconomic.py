

"""
author: Jingru Gong
This file including preprocessing raw data for socio economic data, with thousands indicators
dimension reduction applied, and missing values are moved
visualization shows how different indicators' values changes over time from 2001 to 2014 comparing with the migration rate

### Data pre-processing for Political/socioeconomic data
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
from pandas.core.common import SettingWithCopyWarning


warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

def preprocess():
  socio = pd.read_csv("socioeconomic.csv")
  del socio[socio.columns[0]]
  #melt data format
  socio_long = pd.melt(socio, id_vars = 'Indicator Name', value_vars=['1960', '1961', '1962', '1963', '1964', '1965',
       '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974',
       '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983',
       '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992',
       '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
       '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
       '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019',
       '2020', '2021'])
  socio_long= socio_long.rename(columns = {'variable':'year','Indicator Name':'Indicator_Name'})
  socio_long=socio_long.drop_duplicates(['Indicator_Name','year'],keep= 'last')
  #remove missing values
  missing = socio_long.groupby('Indicator_Name').count().rsub(socio_long.groupby('Indicator_Name', sort=True).size(), axis=0)
  missing.sort_values('value',ascending=False)
  missing = list(missing[missing['value']==0].index)
  socio_long = socio_long[socio_long.Indicator_Name.isin(missing)]
  #to match immigration rate data

  socio_long = socio_long[socio_long["year"].apply(int).between(2001,2021)]
  socio_long['year'] = socio_long['year'].apply(str)

  #scaling
  scaler = MinMaxScaler(feature_range=(10, 100))
  scaler.fit_transform(socio_long['value'].values[:, None])
  
  #dimension reduction
  cols = [
  "GNI (current US$)",
  "Population, total",
  "GDP (current US$)"]
  #remove duplicate and irrelavant indicators
  socio_long = socio_long[socio_long["Indicator_Name"].isin(cols)]
  #deal with some extreme values
  socio_long['value'][socio_long.Indicator_Name == "Population, total"] = socio_long['value']/(10**7)
  socio_long['value'][socio_long.Indicator_Name == "GNI (current US$)"] = socio_long['value']/(10**9)
  socio_long['value'][socio_long.Indicator_Name == "GDP (current US$)"] = socio_long['value']/(10**9)
  #socio_long['value'][socio_long.Indicator_Name == "Agriculture, forestry, and fishing, value added (/%/ of GDP)"] = socio_long['value']/10
  #socio_long['value'][socio_long.Indicator_Name == "Imports of goods and services (/%/ of GDP)"] = socio_long['value']/10
  #socio_long['value'][socio_long.Indicator_Name == "Merchandise trade (/%/ of GDP)"] = socio_long['value']/(10**11)
  #socio_long['value'][socio_long.Indicator_Name == "Exports of goods and services (/%/ of GDP)"] = socio_long['value']/(10**10)

  return socio_long

"""### Visualization"""

def import_migration():
  migration = pd.read_csv("migration.csv")
  migration = migration.rename(columns = {'Date':'year','Net Migration Rate':'Net_Migration_Rate'})
 
  #select only date and migration rate
  migration = migration[['year', 'Net_Migration_Rate']]
  #convert date to year
  migration.year = '20'+migration['year'].apply(lambda x: x.split('/')[-1])
  #to make it consistent with socio data
  migration['Indicator_Name'] = "Net_Migration_Rate"
  migration = migration.rename(columns = {'Net_Migration_Rate':'value'})
  migration['value'] = migration['value'] /10
  return migration

def visualize(socio):
  migration = import_migration()

  plt.figure(figsize=(15,8))
  palette = sns.color_palette("mako_r", 6)
  sns.set_style("darkgrid")
  sns.lineplot(data=socio, x="year", y="value", hue="Indicator_Name", style="Indicator_Name",palette="twilight").set(title='Socioeconomic data with trend in migration rate')
  sns.set_palette("PuBuGn_d")
  pt = sns.lineplot(data=migration, x="year", y="value", palette="rocket",hue="Indicator_Name",style="Indicator_Name",linewidth=2.5)
  pt.legend(title = "Indicator Name")
  pt.figure.savefig('generated_plot/socioeconomic.png')

  
  
  

