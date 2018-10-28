import pandas as pd
import os,logging, functools
from datetime import datetime

logger = logging.Logger('catch_all')

# get OS Path
def get_path(folder,file):
      return os.path.join(folder,file)

# read CSV File using pandas
def read_csv(path):
    # Load the energy dataset
    try:
        #df = pd.read_csv(path, error_bad_lines = False, sep = ';',engine='python', warn_bad_lines = True, na_values=[' ','-'], decimal=",", thousands = ".")
        df = pd.read_csv(path, error_bad_lines = False, sep = ';',engine='python', warn_bad_lines = True, na_values=[' ','-'], decimal=".", thousands = ",", parse_dates=[0])
        print("dataset has {} samples with {} features each.".format(*df.shape))
        # rename ï»¿Date label to Date
        df.rename(columns={'ï»¿Date': 'Date'}, inplace=True)
        
    except Exception as e:
        logger.error(e, exc_info=True)
        print("Dataset could not be loaded. Is the dataset missing?")
    
    #drop columns with NaNs
    """print("Drop column %s " %df.count().idxmin())
    df.drop([df.count().idxmin()],1, inplace=True)"""
    return df

# read multiple CSV files and return a list of data frames
def read_multiple_csv(folder_path,file_list):
      frames = []
      for file in file_list:
            frames.append(read_csv(get_path(folder_path,file)))
      return frames

# Remove all rows with reference to quarterly time values (15,30 and 45) and retain only hourly row information
# time_of_day column has been statically defined
def convert_to_hourly(df):
    allowed_hours = set(['12:00 AM', '1:00 AM', '2:00 AM', '3:00 AM', '4:00 AM', '5:00 AM', '6:00 AM',
       '7:00 AM', '8:00 AM', '9:00 AM', '10:00 AM', '11:00 AM', '12:00 PM', '1:00 PM',
       '2:00 PM', '3:00 PM', '4:00 PM', '5:00 PM', '6:00 PM', '7:00 PM', '8:00 PM',
       '9:00 PM', '10:00 PM', '11:00 PM'])
    #get all unique times available in the dataset. Relevant Feature "Time of day"
    curr_df_hours = set(df.time_of_day.unique())
    excluding_hours = curr_df_hours - allowed_hours
    # delete rows that have quarterly time values 15,30 and 45
    for hour in excluding_hours:
        df = df.drop(df[df.time_of_day==hour].index)
    
    # drop old index and reindex
    df = df.reset_index(drop=True)
    print("Modified dataset has {} samples with {} features each.".format(*df.shape))
    return df

# Perform quarterly to hourly changes to rows of multiple data frames
def convert_multiple_to_hourly(df_list):
      hourly_frames = []
      for df in df_list:
            hourly_frames.append(convert_to_hourly(df))
      return hourly_frames

# Join Data frames based on hourly time values
def join(df_list):
      try:
            # use functools.partial for merge parameters
            merge = functools.partial(pd.merge, on=['date','time_of_day'])
            # use functools.reduce to execute merge iteratively
            result = functools.reduce(merge,df_list)
            
      except Exception as e:
            logger.error(e, exc_info=True)
            print("Merge not possible")
      
      return result

# Get columns with x% NaN
def get_nan_columns(df, threshold_percent):
      row_count = df['Datum'].count()
      #print(row_count)
      min_threshold = int((threshold_percent * 0.01) * row_count)
      is_above_threshold = []

      cols = df.isnull().sum(axis = 0)
      for key,value in cols.iteritems():
            if(value > min_threshold):
                  is_above_threshold.append(key)
      #print("%d columns can be dropped" %len(is_above_threshold))
      return is_above_threshold

def create_date_time(df):
    
    for idx,row in df.iterrows():
        df.set_value(idx,'date_time',datetime.strptime((str(row['Datum']).replace(' 00:00:00', '') + ' ' + row['Uhrzeit'] + ':00'), '%Y-%m-%d %H:%M:%S'))
        return df


                     
# Get column with max NaNs
#df.count().idxmin()