import pandas as pd
from hydroquebec.api import Hydro_quebec_data


def power_api_call(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches wind generation data from the Hydro Quebec API for the given date range.

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: DataFrame containing hourly wind power data.
        """ 
        api_key = 'API_key_here'
        data_type = 'generation'
        data_frame = Hydro_quebec_data(api_key, data_type, start_date, end_date)

        ## WARNING: retrived data_frame is assumed to be a dictionary with keys as the relevant columns
        data_frame = pd.DataFrame(data_frame)  
        return data_frame