## Folder Structure

- `windfarm_weather_data/`: Folder containing a file each for weather data from planned wind-farms (not used). Originally contained all the weather files. 
- `windfarm_weather_data_in_service/`: Folder containing a file each for weather data of **in-service** wind-farms.
- `extract_n_modify_inservice_weather_files.ipynb`: Preprocesses the weather files in the necessary format, assigns labels to weather features based on wind-farms, and moves processed files to Folder `windfarm_weather_data_in_service`.
- `merge_dataframes.ipynb`: Jupyter notebook used to merge the weather and wind power generation data into two main dataframes: one for training/validation, and one for final testing.
- `historique-production-electricite-quebec.csv`: Downloaded csv file containing the power generation data from Hydro-Quebec's website.
- `hydroquebec_wind_farms_in_service.csv`: A csv file containing project name, location, and wind-turbine information etc for all the **in-service** windfarms contracted by Hydro-Quebec.
- `hydroquebec_wind_farms.csv`: A csv file containing project name, location, and wind-turbine information etc for all the in-service windfarms contracted by Hydro-Quebec.
- `Wind_Turbine_Database_FGP.xlsx`: File containing various useful information, such as location coordinates, turbine mode, and rotor diameter etc for the windfarms.