"""
Team: A12
            Jingru Gong, Alex Yang, Alice Ji, Swathi Parvathaneni
Andrew id:  jingrug, yihsuehy, jiaxuanj, 
"""
import migration as m
import socioeconomic as socio
import weather_model 
import seismic as s

# get data ready


def get_data():
    migration = m.get_data()
    socioeconomic = socio.preprocess()
    weather = weather_model.get_data()
    seismic = s.get_data()


    return migration,socioeconomic,weather,seismic
    

def main():
    #get the data ready
    
    migration, socioeconomic , weather,seismic = get_data()

    #visualize socio economic data with trend in migration
    print("--------Socioeconomic--------")
    socio.visualize(socioeconomic)
    print('\n')

    #get model results from migration linear regression
    print("--------Migration--------")
    m.get_model_result()
    print('\n')

    #visualize counts of floods or not in weather data after undersampling
    print("--------Weather--------")
    weather_model.visual_rand_over_samp()
    #get the model result of the kNN classifier model
    weather_model.result()
    print('\n')

    #seismic
    print("--------Seismic--------")
    s.main()
    print('\n')
    





if __name__ == "__main__":
    main()
    






