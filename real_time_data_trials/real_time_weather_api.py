import pandas as pd
from hydroquebec.api import Hydro_quebec_data


api_key = 'API_key_here'

data_type = 'generation'

start_date = 'YYYY-MM-DD'
end_date = 'YYYY-MM-DD'

start_date = pd.Timestamp(start_date) - pd.Timedelta(3, unit='y') ## download the data from 3 previous years to train
start_date = start_date.strftime('%y-%m-%d')

data_frame = Hydro_quebec_data(api_key, data_type, start_date, end_date)

data_frame.to_csv("real_time_data/real_time_power_data.csv")