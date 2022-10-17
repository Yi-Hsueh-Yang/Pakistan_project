# -*- coding: utf-8 -*-
"""
Created on Sat Oct 1 10:59:06 2022

@author: swathi

The following script scrapes data from https://seismic.pmd.gov.pk/events.php 
which has seismic events from 2016 in the Asia- Pacific region.
 This data has been used to predict Magnitude and Depth of earthquakes respective to region using Simple Regression Model. 
 Further we visualized trends of the magnitude and Depth of earthquakes over the years and 
 plotted the most affected regions due to earthquakes in Pakistan as a Heatmap using gmaps.
"""

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from random import randint
from time import sleep
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gmaps 
import gmaps.datasets 
from ipywidgets.embed import embed_minimal_html
from datetime import datetime
import time
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

#scrapes data from url and pre-prossess it for visualization and prediction
#This function takes hours to run , so we will recommend use the cleaned data instead
def scrape_pmd_seismic_data(url):
    print("\n----Scraping data from: ",url,"----\n")
    seismic_data=[['Date','Time','Latitude','Longitude','Magnitude','Depth(km)','Region']]
    for page in tqdm(range(1,291)):
        request_site = Request(url+'?page='+str(page)+'/', headers={"User-Agent": "Mozilla/5.0"})
        webpage = urlopen(request_site).read()
        bsyc = BeautifulSoup(webpage, "lxml")
        table_list = bsyc.findAll('table')
        table = table_list[0]
        rows = table.findAll('tr')
        for row in rows[1:]:
            row_data = row.findAll('td')
            seismic_data.append([x.contents[0] for x in row_data[0:-2]])
        sleep(randint(2,10))
        
    print("Sucessfully scraped data from: ",url)
    #converting webscrapped data to a dataframe
    df = pd.DataFrame(seismic_data, columns = seismic_data[0])

    #Data pre-processing
    #clean latitude - removing alphabets and special charecters, extra spaces and changing dtype to float
    df["Latitude"] = df["Latitude"].replace(to_replace=r'\.\.',value ='.', regex=True)
    df["Latitude"] = df["Latitude"].replace(to_replace=r'[a-zA-Z\s`]',value = '', regex=True).str.strip()
    df = df[df["Latitude"].str.contains(r'^\d+\.\d+$')]
    df["Latitude"] = df["Latitude"].astype(float)
    #clean longitude - removing alphabets and special charecters, extra spaces and changing dtype to float
    df["Longitude"] = df["Longitude"].replace(to_replace=r'\.\.',value ='.', regex=True)
    df["Longitude"] = df["Longitude"].replace(to_replace=r'[a-zA-Z\s`]',value = '', regex=True).str.strip() 
    df = df[df["Longitude"].str.contains(r'^\d+\.\d+$')]
    df["Longitude"] = df["Longitude"].astype(float)
    df.to_csv('cleaned_seismic_data.csv',index=False)
    print("Saved pre-processed data to cleaned_seismic_data.csv")
    return df

def get_data():
    df = pd.read_csv("seismic.csv")
    return df
#Data visualizations for the Pakistan data
def genrate_seismic_data_plots(df):
    print("\n\n----generating visualizations for Pakistan data----")
    #filtering seismic events related to Pakistan
    pak_identifiers = ["pakistan", "pk","Balochistan","Khyber","Punjab","Sindh","Islamabad"]
    pak_df = df[df["Region"].str.contains(r""+'|'.join(pak_identifiers), case = False)]
    pak_df['Timestamp']= pd.to_datetime(df['Date'] +' '+ df['Time'], format='%d/%m/%Y %H:%M:%S')
    pak_df = pak_df.drop(['Date','Time'], axis=1)
    #writing Pakistan data to file
    
    pak_df.to_csv('pak_data.csv')

    #time series for Magnitude of Seismic events occured in Pak from 2016-2022
    dims = (15, 10)
    fig, ax = plt.subplots(figsize=dims)
    sns.lineplot(x = "Timestamp", y = "Magnitude", data = pak_df)
    plt.xlabel("Year")
    plt.ylabel("Magnitude")
    plt.title("Magnitude of earth quakes in Pakistan since 2016")
    plt.xticks(rotation = 20)
    plt.savefig('generated_plot/PK_earthquakes_timeseries_magnitude.png')
    print("Displaying visualization Magnitude of earth quakes in Pakistan since 2016")
    plt.show()
    
    #time series for Depth of earth quakes occured in Pakistan between 2016-2022
    fig, ax = plt.subplots(figsize=dims)
    sns.lineplot(x = "Timestamp", y = "Depth(km)", data = pak_df)
    plt.xlabel("Year")
    plt.ylabel("Depth(km)")
    plt.title("Depth(km) of earth quakes in Pakistan")
    plt.xticks(rotation = 20)
    plt.savefig('generated_plot/PK_earthquakes_timeseries_depth.png')
    print("Displaying Depth(km) of earth quakes in Pakistan since 2016")
    plt.show()


    #Histogram of seismic events recorded per year in Pakistan
    fig, ax = plt.subplots(figsize=dims)
    pak_df['year'] = pd.DatetimeIndex(pak_df['Timestamp']).year
    sns.histplot(data=pak_df, x="year")
    plt.title("No. of earth quakes per year in Pakistan")
    plt.savefig('generated_plot/PK_earthquakes_histogram.png')
    print("Displaying No. of earth quakes per year in Pakistan since 2016")
    plt.show()


    #select range of cordinates
    # Pakistan lies between 23 degrees 35 minutes to 37 degrees 05 minutes North latitude
    # 60 degrees 50 minutes to 77 degrees 50 minutes east longitude.
    #divinding the longitude and latitude range in 100 equal intervals
    Lat = np.arange(23.35, 37.05, 0.137) 
    Lon = np.arange(60.50, 77.50, 0.17) 
    longitude_values = [Lon,]*100 
    latitude_values = np.repeat(Lat,100) 

    #creating 100x100 grid for recorded events
    event_counts = np.zeros((100,100))
    event_counts = np.zeros((100,100))
    for a in range(len(pak_df)):
        for b1 in range(100):
            if Lat[b1]-0.137<=pak_df['Latitude'].values[a]<Lat[b1]+0.137:
                for b2 in range(100):
                      if Lon[b2]-0.17<=pak_df['Longitude'].values[a]<Lon[b2]+0.17:
                            event_counts[b1,b2] += 1
    event_counts.resize((10000,))

    #Consolidating heatmap data
    heatmap_data = {'Counts': event_counts, 'latitude': latitude_values, 'longitude' : np.concatenate(longitude_values)} 
    map_df = pd.DataFrame(data=heatmap_data) 
    filtered_map_df = map_df[map_df['Counts']>0]
    locations = filtered_map_df[['latitude', 'longitude']] 
    weights = filtered_map_df['Counts']
    filtered_map_df.head()


    #configuring gmap layout 
    figure_layout = {
        'width': '100%',
        'height': '120vh',
        'border': '2px solid white',
        'map_type': 'TERRAIN',
        'padding': '2px'
    }
    # seting center coordinates (the geographic center of Pak 29°59'53"N   69°59'57"E)
    center_coordinates = (29.59, 69.59)
    #configuring gmaps key
    gmaps.configure(api_key='AIzaSyBIgOdPC6sot9CuuKGuLdpOqcLtPJ0o_x4')
    #plotting
    fig = gmaps.figure(center=center_coordinates, 
                       zoom_level=6, 
                       layout=figure_layout)
    heatmap_layer = gmaps.heatmap_layer(locations, weights=weights, max_intensity=5)
    fig.add_layer(heatmap_layer) 
    embed_minimal_html('generated_plot/PK_earthquakes_heatmap.html', views=[fig])
    print("Saved map showing most affected regions in Pakistan to: ./generated_plot/PK_earthquakes_heatmap.html")

#model predicts magnitude and depth of an earthquake
def regression__model_generator(df):
    #Scaling date time to unix time for model
    print("\n\n----Generating data for Regression model----")
    timestamps = []
    for d, t in zip(df['Date'], df['Time']):
        try:
            ts = datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
            timestamps.append(time.mktime(ts.timetuple()))
        except ValueError:
            timestamps.append('NA')
    df['Timestamp'] = pd.Series(timestamps).values
    model_df = df.drop(['Date', 'Time'], axis=1)
    model_df = model_df[model_df.Timestamp != 'NA']
    model_df.head()

    #splitting data into train and test
    print("Splitting train and test data")
    X = model_df[['Timestamp', 'Latitude', 'Longitude']]
    y = model_df[['Magnitude', 'Depth(km)']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)
    print("train data dimensions:  ",X_train.shape, y_train.shape)
    print("test data dimensions:  ",X_test.shape, y_test.shape)
    #fitting the model to data
    print("Fitting data for Regressin model")
    reg = LinearRegression().fit(X_train, y_train)
    #predicting Magnitude and depth using model
    print("Predicting for test data")
    y_predict = reg.predict(X_test)
    #evaluation metrics for the model
    model_score = reg.score(X, y)
    mean_absolute_error = metrics.mean_absolute_error(y_test,y_predict)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(y_test,y_predict))
    print("\n\n----Model evaluation----")
    print("model score: ",model_score)
    print("mean absolute_error: ",mean_absolute_error)
    print("root mean squared error: ",root_mean_squared_error)
    #model score:  85.27508
    #mean_absolute_error:  85.27508
    #root mean squared error:  85.27508
    



def main():
    url = "https://seismic.pmd.gov.pk/events.php"
    df = scrape_pmd_seismic_data(url)
    #df = pd.read_csv('seismic.csv') #remove comment above line to scrape data
    genrate_seismic_data_plots(df)
    regression__model_generator(df)

if __name__ == "__main__":
    main()


