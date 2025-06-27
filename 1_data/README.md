## Folder Structure

- `data_collection/`: Folder contains:
    - `data_sources.txt`: Contains the links for websites used to download data.

    - `getWindFarmsFromHQ-checkpoint.ipynb`: Jupyter notebook used to scrape location information for the wind farms used by HydroQuebec from the HydroQeubec website: https://www.hydroquebec.com/electricity-purchases-quebec/electricity-contracts.html. We downloaded information on windfarms located in Quebec, which is made available by the Canadian government at: https://open.canada.ca/data/en/dataset/79fdad93-9025-49ad-ba16-c26d718cc070. 

    - `windfarm_weather_downloader-checkpoint.ipynb`: Jupyter notebook used to download weather data from the open-meteo website.

- `data_preprocessing/`: Folder contains relevant files for data preprocessing - contains its own *README*.

- `final_dataframes/`: Folder contains the final dataframes used for training/validation and testing.
    - `main_testing_dataframe.csv`: Contains our testing data of 2019-2023 (testing year 2023).
    - `main_validation_dataframe.csv`: Contains our training and validation data of 2019-2022.

