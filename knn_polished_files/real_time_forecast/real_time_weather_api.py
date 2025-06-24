import time
import requests
import pandas as pd
from typing import List, Dict, Optional
import os



#Function to slow down requests to avoid rate limit on the API
def fetch_with_retry(url, params, max_retries=3):
    for _ in range(max_retries):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            wait_time = int(response.headers.get("Retry-After", 10))
            print(f"Rate limited. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print(f"Error {response.status_code}: {response.text}")
            break
    return None


def fetch_historical_weather_multiple(
    latitudes: List[float],
    longitudes: List[float],
    start_datetime: str,
    end_datetime: str,
    location_names: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical weather data for multiple lat/lon pairs, including wind direction.

    Args:
        latitudes: List of latitudes.
        longitudes: List of longitudes.
        start_date: Start date in "YYYY-MM-DD" format.
        end_date: End date in "YYYY-MM-DD" format.
        location_names: Optional names for each location (default: "loc_0", "loc_1", ...).

    Returns:
        Dictionary of DataFrames (key: location name, value: weather data).
    """
    start_date = start_datetime.split("T")[0]
    end_date = end_datetime.split("T")[0]

    if len(latitudes) != len(longitudes):
        raise ValueError("Latitudes and longitudes must have the same length.")
    
    if location_names is None:
        location_names = [f"loc_{i}" for i in range(len(latitudes))]
    elif len(location_names) != len(latitudes):
        raise ValueError("Location names must match latitudes/longitudes length.")

    weather_data = {}
    #base_url = "https://archive-api.open-meteo.com/v1/archive"         ## for archived data
    base_url = "https://api.open-meteo.com/v1/forecast"                 ## for forcasted data

    for lat, lon, name in zip(latitudes, longitudes, location_names):
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "temperature_2m",
                "wind_speed_10m",
                "relative_humidity_2m"
            ],
            "timezone": "UTC"
        }
        
        data = fetch_with_retry(base_url, params=params,max_retries=5)

        df = pd.DataFrame(data["hourly"])
        df['time'] = pd.to_datetime(df['time'], utc=True)
        time_filter = (df["time"] >= start_datetime) & (df["time"] <= end_datetime)
        df["location"] = name 
        df = df[time_filter]
        df['time'] = df['time'] - pd.Timedelta(hours=5)
        weather_data[name] = df

    return weather_data


def weather_api_call(start_date, end_date):
    ## Setting up the required varibales to call the above function
    file_path = os.path.join(os.path.dirname(__file__)) + '../../data/raw_data/hydroquebec_wind_farms_in_service.csv'
    position_df = pd.read_csv(file_path)
    # Example: Montreal (45.5017° N, 73.5673° W)
    latitude = position_df['latitude']
    longitude = position_df['longitude']

    start_date += "T05:00:00"     ## adjusting for the time-zone to UTC conversion
    end_date = pd.Timestamp(end_date)+ pd.Timedelta(1,unit='d')
    end_date = end_date.strftime('%Y-%m-%d')     
    end_date += "T04:00:00"

    weather_data = fetch_historical_weather_multiple(latitude, longitude, start_date, 
                                                     end_date,location_names=position_df['name'])

    keys = [x for x in weather_data.keys()]
    merge_wdf = pd.DataFrame()
    for key, label in zip(keys, position_df["labels"]):
        df = pd.DataFrame(weather_data[key])
        
        new_columns = [f"{column}" for column in df.columns[0:1]] + [f"{column}_{label}" for column in df.columns[1:]]
        df.columns = new_columns
        df = df.drop(columns=[new_columns[-1]])                    ## drop the location, no need anymore
        df['time'] = pd.to_datetime(df['time'])                 ## convert time to pandas timestamp
        df = df.sort_values('time')   
        if key == keys[0]:
            merge_wdf = df
        else:
            merge_wdf = pd.merge(merge_wdf, df, on=['time'])
    merge_wdf = merge_wdf.dropna()
    return merge_wdf

