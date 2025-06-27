## Folder Structure

- `data_collection/`: Folder contains:
    - `data_sources.txt`: Contains the links for websites used to download data.

    - `getWindFarmsFromHQ-checkpoint.ipynb`: Jupyter notebook used to scrape location information for the wind farms used by HydroQuebec from the HydroQeubec website: https://www.hydroquebec.com/electricity-purchases-quebec/electricity-contracts.html. We downloaded information on windfarms located in Quebec, which is made available by the Canadian government at: https://open.canada.ca/data/en/dataset/79fdad93-9025-49ad-ba16-c26d718cc070. 

    - `windfarm_weather_downloader-checkpoint.ipynb`: Jupyter notebook used to download weather data from the open-meteo website.

- `data_preprocessing/`: Folder contains:
    - `windfarm_weather_data/`: Folder containing a file each for weather data from planned wind-farms (not used). Originally contained all the weather files. 
    - `windfarm_weather_data_in_service/`: Folder containing a file each for weather data of **in-service** wind-farms.
    - `extract_n_modify_inservice_weather_files.ipynb`: Preprocesses the weather files in the necessary format, assigns labels to weather features based on wind-farms, and moves processed files to Folder `windfarm_weather_data_in_service`.
    - `merge_dataframes.ipynb`: Jupyter notebook used to merge the weather and wind power generation data into two main dataframes: one for training/validation, and one for final testing.
    - `historique-production-electricite-quebec.csv`: Downloaded csv file containing the power generation data from Hydro-Quebec's website.
    - `hydroquebec_wind_farms_in_service.csv`: A csv file containing project name, location, and wind-turbine information etc for all the **in-service** windfarms contracted by Hydro-Quebec.
    - `hydroquebec_wind_farms.csv`: A csv file containing project name, location, and wind-turbine information etc for all the in-service windfarms contracted by Hydro-Quebec.
    - `Wind_Turbine_Database_FGP.xlsx`: File containing various useful information, such as location coordinates, turbine mode, and rotor diameter etc for the windfarms.

- `final_dataframes/`: Folder contains the final dataframes used for training/validation and testing.
    - `main_testing_dataframe.csv`: Contains our testing data of 2019-2023 (testing year 2023).
    - `main_validation_dataframe.csv`: Contains our training and validation data of 2019-2022.

