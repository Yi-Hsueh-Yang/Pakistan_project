import tabula.io as tb
import pandas as pd
import glob

months_data = ['April_2021.pdf', 'August_2021.pdf', 'December_2021.pdf', 'February_2021.pdf', 'January_2021.pdf', 'July_2021.pdf', 'June_2021.pdf', 'March_2021.pdf', 'May_2021.pdf', 'November_2021.pdf', 'October_2021.pdf', 'September_2021.pdf']        




def scrap_pdf(months_data):
    RowasHeaderData = pd.DataFrame()
    semiCleanedData = pd.DataFrame()
    RowDroppedData = pd.DataFrame() 
    dataReady = pd.DataFrame()

    path = '/weather_pdf/'
    #read in pdf files and scrape it with tb.read_pdf
    
    for month in months_data:
        #print(month)
        if month == 'April_2021.pdf': 
            data = tb.read_pdf(path + month, area = (55,61.45,710,550), pages=5, pandas_options={'header':None}, stream="True", output_format="dataframe")[0] #make 'data' a dataframe
            #fill all the NAs in the 'station' column with different station name, each station should have 2 different records
            for i in range(len(data.loc[:,0])-1):
                if pd.isna(data.loc[:,0][i]):
                    data.loc[:,0][i] = data.loc[:,0][i+1]
            for i in range(len(data.loc[:,0])-1):
                if pd.isna(data.loc[:,0][i]):
                    data.loc[:,0][i] = data.loc[:,0][i-1]

            #drop first two rows which are full of NAs, incomplete column names but with no actual data
            data = data.iloc[2: , : ]
            data.iloc[-1,0] = 'Karachi'

            #Move first row of data to header
            data.columns = data.iloc[0]
            RowasHeaderData = data.drop(data.index[0])

            #Rename all columns
            semiCleanedData = RowasHeaderData.rename(columns={"Station Level Sea Level Mean":"PressureStationLevel_PressureSeaLevel_MeanTemp",
                                    "Maximum Minimum":"MaxTemp_MinTemp", "Pressure":"VapourPressure(mb)",
                                    "Total (mm) Days":"PercipitationTotal"})

            #Drop rows with NAs
            RowDroppedData = semiCleanedData.dropna()

            #split columns with 2 different data into 2 different columns
            RowDroppedData[['PressureStationLevel','PressureSeaLevel','MeanTemp']] = RowDroppedData.PressureStationLevel_PressureSeaLevel_MeanTemp.str.split(" ", expand=True)
            RowDroppedData[['MaxTemp','MinTemp']] = RowDroppedData.MaxTemp_MinTemp.str.split(" ", expand=True)
            RowDroppedData[['PercipitationTotal','PercipitationDays']] = RowDroppedData.PercipitationTotal.str.split(" ", expand=True)

            #transform all the data into integers except the first station column
            cols = RowDroppedData.columns
            RowDroppedData[cols[1:]] = RowDroppedData[cols[1:]].apply(pd.to_numeric, errors='coerce')

            #Drop those columns consist of 2 data
            dataReady = RowDroppedData.drop(columns={'PressureStationLevel_PressureSeaLevel_MeanTemp','MaxTemp_MinTemp'})    
            dataReady.to_csv(month+'.csv')

        elif month in ['August_2021.pdf', 'March_2021.pdf']:
            data = tb.read_pdf(path + month, area = (55,61.45,710,550), pages=5, pandas_options={'header':None}, stream="True", output_format="dataframe")[0]
            for i in range(len(data.loc[:,0])-1):
                if pd.isna(data.loc[:,0][i]):
                    data.loc[:,0][i] = data.loc[:,0][i+1]
            for i in range(len(data.loc[:,0])-1):
                if pd.isna(data.loc[:,0][i]):
                    data.loc[:,0][i] = data.loc[:,0][i-1]
            data = data.iloc[2: , : ]
            data.iloc[-1,0] = 'Karachi'
            data.columns = data.iloc[0]
            RowasHeaderData = data.drop(data.index[0])
            semiCleanedData = RowasHeaderData.rename(columns={"Station Level Sea Level":"PressureStationLevel_PressureSeaLevel_MeanTemp",
                                    "Mean":"MeanTemp", "Maximum Minimum":"MaxTemp_MinTemp", "Pressure":"VapourPressure(mb)",
                                    "Total (mm) Days":"PercipitationTotal"})
            RowDroppedData = semiCleanedData.dropna()
            RowDroppedData[['PressureStationLevel','PressureSeaLevel']] = RowDroppedData.PressureStationLevel_PressureSeaLevel_MeanTemp.str.split(" ", expand=True)
            RowDroppedData[['MaxTemp','MinTemp']] = RowDroppedData.MaxTemp_MinTemp.str.split(" ", expand=True)
            RowDroppedData[['PercipitationTotal','PercipitationDays']] = RowDroppedData.PercipitationTotal.str.split(" ", expand=True)
            cols = RowDroppedData.columns
            RowDroppedData[cols[1:]] = RowDroppedData[cols[1:]].apply(pd.to_numeric, errors='coerce')                
            dataReady = RowDroppedData.drop(columns={'PressureStationLevel_PressureSeaLevel_MeanTemp','MaxTemp_MinTemp'})
            dataReady.to_csv(month+'.csv')   

        elif month in ['December_2021.pdf', 'January_2021.pdf', 'July_2021.pdf', 'June_2021.pdf', 'November_2021.pdf','September_2021']:
            data = tb.read_pdf(path + month, area = (55,61.45,700,550), pages=5, pandas_options={'header':None}, stream="True", output_format="dataframe")[0]
            for i in range(len(data.loc[:,0])-1):
                if pd.isna(data.loc[:,0][i]):
                    data.loc[:,0][i] = data.loc[:,0][i+1]
            for i in range(len(data.loc[:,0])-1):
                if pd.isna(data.loc[:,0][i]):
                    data.loc[:,0][i] = data.loc[:,0][i-1]
            data = data.iloc[2: , : ]
            data.iloc[-1,0] = 'Karachi'
            data.columns = data.iloc[0]
            RowasHeaderData = data.drop(data.index[0])
            semiCleanedData = RowasHeaderData.rename(columns={"Station Level Sea Level":"PressureStationLevel_PressureSeaLevel_MeanTemp",
                                    "Mean":"MeanTemp", "Maximum Minimum":"MaxTemp_MinTemp", "Pressure":"VapourPressure(mb)",
                                    "Total (mm) Days":"PercipitationTotal"})
            RowDroppedData = semiCleanedData.dropna()
            RowDroppedData[['PressureStationLevel','PressureSeaLevel']] = RowDroppedData.PressureStationLevel_PressureSeaLevel_MeanTemp.str.split(" ", expand=True)
            RowDroppedData[['MaxTemp','MinTemp']] = RowDroppedData.MaxTemp_MinTemp.str.split(" ", expand=True)
            RowDroppedData[['PercipitationTotal','PercipitationDays']] = RowDroppedData.PercipitationTotal.str.split(" ", expand=True)
            cols = RowDroppedData.columns
            RowDroppedData[cols[1:]] = RowDroppedData[cols[1:]].apply(pd.to_numeric, errors='coerce')                
            dataReady = RowDroppedData.drop(columns={'PressureStationLevel_PressureSeaLevel_MeanTemp','MaxTemp_MinTemp'})
            dataReady.to_csv(month+'.csv')
                    
scrap_pdf(months_data)