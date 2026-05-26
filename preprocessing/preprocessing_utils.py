import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from imblearn.pipeline import Pipeline
import re
from astropy.time import Time
from scipy.signal import periodogram
import os
from scipy.signal import periodogram, find_peaks, find_peaks_cwt
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.seasonal import STL
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import timeit
import cProfile
import logging

class OMNIPreprocessor():

    def safe_gradient(self, v):
        if len(v) > 1:
            return np.gradient(v)
        else:
            return np.zeros_like(v)
            
    def calculate_position_anomalies(self, df, id_type="IMF", gradient_thr=0.5):
        L1_pos = (235.441845864, 0, 0)
        id_column = f'Spacecraft ID ({id_type})'
        distance_column = 'Approximate Distance to SEL (Re)'
        position_columns = ['Spacecraft Position x', 'Spacecraft Position y', 'Spacecraft Position z']

        df['Approximate Distance to SEL (Re)'] = np.sqrt(   (df[position_columns[0]] - L1_pos[0])**2 + 
                                                            (df[position_columns[1]] - L1_pos[1])**2 + 
                                                            (df[position_columns[2]] - L1_pos[2])**2)   
        
        df['L1 Distance Gradient'] = df.groupby(id_column)[distance_column].transform(lambda v: self.safe_gradient(v))
        df['L1 Distance Gradient Residuals'] = df.groupby(id_column)['L1 Distance Gradient'].transform(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        df['Position Anomaly'] = 0
        
        # Flagging anomalies based on gradient threshold
        df.loc[df['L1 Distance Gradient'] > gradient_thr, 'Position Anomaly'] = 20
        # Rolling max to smooth the anomalies
        df['Position Anomaly'] = df.groupby(id_column)['Position Anomaly'].transform(lambda v: v.rolling(window=3, center=True).max())
        df['Position Anomaly'] = df.groupby(id_column)['Position Anomaly'].transform(lambda v: v.interpolate(method='nearest')).ffill().bfill()
        # Handle anomalies by setting distance to NaN
        df.loc[df['Position Anomaly'] == 20, distance_column] = np.nan
        df.loc[df['Position Anomaly'] == 20, position_columns] = np.nan

        # Interpolate to fill NaN values
        df[position_columns] = df.groupby(id_column)[position_columns].transform(lambda v: v.interpolate(method='nearest'))

        df[distance_column] = np.sqrt(  (df[position_columns[0]] - L1_pos[0])**2 + 
                                        (df[position_columns[1]] - L1_pos[1])**2 + 
                                        (df[position_columns[2]] - L1_pos[2])**2)   
        return df

    def add_additional_features(self, df):
        start_date = df.index.min().date()
        df['Days Since Start'] = (df.index.date - start_date)
        df['Days Since Start'] = df['Days Since Start'].apply(lambda x: x.days)
        df['Bartels Rotation Number'] = ((df['Days Since Start']+16) // 27) + 2204
        df.drop(columns='Days Since Start', inplace=True)

        df['Solar Cycle'] = df.index.map(lambda x: (x.year - 2008) // 11 + 24)
        df['Bartels Rotation Number'] = df['Bartels Rotation Number'].ffill()
        df['Solar Cycle'] = df['Solar Cycle'].ffill()
        return df

    def calculate_slopes_noncentered(self, df, window_size, cadence, day_interval, columns):
        for _, col in enumerate(columns):
            for i in range(window_size, len(df)):
                window_data = df.iloc[(i - window_size) : i]

                x = window_data.reset_index().index.values * (cadence / day_interval)
                y = window_data[col].values
                slope, intercept = np.polyfit(x, y, 1)

                fit_line = slope * x + intercept
                residuals = y - fit_line
                std_residual = np.std(residuals)
                
                x_mean = np.mean(x)
                spread_x = np.sum((x - x_mean)**2)

                se_slope = std_residual / np.sqrt(spread_x)

                df.at[df.index[i], f'{col}_res_std'] = std_residual
                df.at[df.index[i], f'{col}_slope'] = slope
                df.at[df.index[i], f'{col}_intercept'] = intercept
                df.at[df.index[i], f'{col}_se_slope'] = se_slope
                df.at[df.index[i], f'{col}_res'] = y[(window_size//2)] - (slope * x[(window_size//2)] + intercept)

        df = df.iloc[window_size:].copy()
        return df

    def calculate_slopes(self, df, window_size, cadence, day_interval, columns):
        for _, col in enumerate(columns):
            for i in range(window_size//2, len(df)-(window_size//2)):
                window_data = df.iloc[i - (window_size//2) : i + (window_size//2)]

                x = window_data.reset_index().index.values * (cadence / day_interval)
                y = window_data[col].values
                slope, intercept = np.polyfit(x, y, 1)

                fit_line = slope * x + intercept
                residuals = y - fit_line
                std_residual = np.std(residuals)
                
                x_mean = np.mean(x)
                spread_x = np.sum((x - x_mean)**2)

                se_slope = std_residual / np.sqrt(spread_x)

                df.at[df.index[i], f'{col}_res_std'] = std_residual
                df.at[df.index[i], f'{col}_slope'] = slope
                df.at[df.index[i], f'{col}_intercept'] = intercept
                df.at[df.index[i], f'{col}_se_slope'] = se_slope
                df.at[df.index[i], f'{col}_res'] = y[(window_size//2)] - (slope * x[(window_size//2)] + intercept)

        df = df.iloc[(window_size//2):-(window_size//2)].copy()
        return df
    
    def parallel_slope_calculation(self, dataframe, window_size, cadence, day_interval, columns, num_processes=8, centered=True):
        chunk_size = len(dataframe) // num_processes
        chunks = [dataframe[i:i + chunk_size].copy() for i in range(0, len(dataframe), chunk_size - window_size)]

        if len(chunks[-1]) < chunk_size:
            last_chunk = pd.concat([chunks[-2], chunks[-1]])
            last_chunk = last_chunk[~last_chunk.index.duplicated(keep='first')]
            chunks = chunks[:-2] + [last_chunk]

        col_ext = ['_res_std', '_slope', '_intercept', '_se_slope', '_res']
        col_ext = [f'{col}{ext}' for col in columns for ext in col_ext]
        dataframe[col_ext] = np.nan

        with mp.Pool(processes=num_processes) as pool:
            processed_chunks = pool.starmap(self.calculate_slopes if centered else self.calculate_slopes_noncentered, [(chunk, window_size, cadence, day_interval, columns) for chunk in chunks])

        final_df = pd.concat([dataframe[:window_size//2], *processed_chunks, dataframe[-(window_size//2):]])
        final_df.sort_index(inplace=True)

        final_df[col_ext] = final_df[col_ext].interpolate(method='linear', limit_direction='both')

        print(f"Duplicate indices: {final_df.index.duplicated().sum()}")
        print(f"Length of final df: {len(final_df)}, Length of original df: {len(dataframe)}")

        return final_df

    def add_slopes(self, df, columns, centered=True):
        cadence=30
        hour_interval = 60*60
        hour_count = 24
        day_count=1
        day_interval = (day_count*hour_count*hour_interval)
        window_size= day_interval // cadence

        df = self.parallel_slope_calculation(df, window_size, cadence, day_interval, columns, centered=centered)
        return df

def process_file(file_path):
    df = pd.read_fwf(file_path, colspecs='infer', skiprows=get_skiprows(file_path))
    return df

def get_skiprows(file_path):
    """Determine how many rows to skip to reach the data."""
    with open(file_path, 'r') as f:
        skiprows = 0
        while f.readline() != '\n':
            skiprows += 1
    return skiprows
    
class PODPreprocessor():
    def process_chunk(self, args):
        logger = logging.getLogger(__name__)

        i, chunk = args
        logger.info(f"Processing chunk {i+1}...")
        logger.info(f"{chunk.index.min()} - {chunk.index.max()}")
        return self.compute_decomposition(chunk, 'a [m]')
    
    def split_dataframe_into_chunks(self, df):
        logger = logging.getLogger(__name__)
        # Split df into chunks of 14 days (assuming 2-minute intervals)
        chunk_size = 14 * 24 * 60 * 2
        chunks = [df[i:i+chunk_size].copy() for i in range(0, len(df), chunk_size)]
        logger.info("Chunks for decomposition: ")
        [logger.info(f'{(chunk.index[0], chunk.index[-1])}\n') for chunk in chunks]
        
        # Check if the last entry is less than 14 days and concatenate if needed
        if len(chunks[-1]) < chunk_size:
            last_chunk = pd.concat([chunks[-2], chunks[-1]])
            chunks = chunks[:-2] + [last_chunk]
        return chunks
    
    def parallel_process(self, df, num_processes=8):

        logger = logging.getLogger(__name__)
        chunks = self.split_dataframe_into_chunks(df)

        args = [(i, chunk) for i, chunk in enumerate(chunks)]
        
        with mp.Pool(processes=num_processes) as pool:
            processed_chunks = list(tqdm(pool.imap_unordered(self.process_chunk, args), total=len(args)))
            logger.info("Finished processing chunks. Merging results.")
            final_df = pd.concat(processed_chunks)

        logger.info("Merged.")
        
        return final_df

    def preprocess_pod(self, lst_path, out_path, cadence='30s'):
        cfg = json.load(
            open(f"{lst_path.replace('raw', 'configs')}/config.json"))
        satellite_names = cfg.keys()
        logger = logging.getLogger(__name__)

        for satellite_name in satellite_names:
            logger.info(f"Preprocessing {satellite_name}... Started at time {timeit.default_timer()}.")

            df = self.process_pod_data_parallel_v2(lst_path, satellite_name)
            df.sort_index(inplace=True)

            missing_days = []
            logger.info(f"Missing days for {satellite_name}:")
            for i in range(len(df.index) - 1):
                if df.index[i + 1] - df.index[i] > pd.Timedelta(hours=1):
                    logger.info(f"Missing day between {df.index[i]} and {df.index[i + 1]}")
                    logger.info(f"Missing day doy: {(df.index[i]+pd.Timedelta(days=1)).dayofyear}")
                    missing_days.append(df.index[i])

            logger.info(f"Finished Reading LST files for {satellite_name} at time {timeit.default_timer()}.")
           
            profiler = cProfile.Profile()
            profiler.enable()
            df = self.parallel_process(df)
            profiler.disable()
            profiler.dump_stats(f"decomposition_profile_output_chunk.txt")
            logger.info(f"Finished Computing Signal Decomposition for {satellite_name} at time {timeit.default_timer()}.")
            logger.info ("Last date processed: ", df.index[-1])
            df.sort_index(inplace=True)

            profiler = cProfile.Profile()
            profiler.enable()
            df = self.add_od_features(df)
            df.sort_index(inplace=True)

            profiler.disable()
            profiler.dump_stats(f"od_profile_output_chunk.txt")
            logger.info(f"Finished Computing Orbital Decay for {satellite_name} at time {timeit.default_timer()}.")
            logger.info ("Last date processed: ", df.index[-1])

            df.to_parquet(f"{out_path}/{satellite_name}.csv",
                      index=True, index_label='time')

    def get_regression_coefficients(self, x, y):
        x = np.array(x)
        y = np.array(y)
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, b

    def filter_by_log_intervals(self, values):
        first_in_interval = {}
        
        for value in values:
            log_interval = int(np.floor(np.log10(value)))

            # ignore if > 10^3
            if log_interval >= 3:    
                continue
            
            if log_interval not in first_in_interval:
                first_in_interval[log_interval] = value
        
        return list(first_in_interval.values())

    def get_period(self, dataframe, column, detrend, already_processed, cadence='30s'):
        logger = logging.getLogger(__name__)

        fs = (dataframe.index.max() - dataframe.index.min()) // pd.Timedelta(cadence)
        # fs = 1 / pd.Timedelta("30s").total_seconds()  # = 1 / 30 Hz = 0.0333 Hz

        freqencies, spectrum = periodogram(
            dataframe[column],
            fs=fs,
            detrend=detrend,
            window="boxcar",
            scaling='spectrum'
        )

        sorted_spectra_idx = np.argsort(spectrum)[::-1]
        sorted_freqs = freqencies[sorted_spectra_idx]
        sorted_freqs = sorted_freqs[sorted_freqs > 10]
        periods = [round (fs / freq) for freq in sorted_freqs]
        periods = list(dict.fromkeys(periods))

        unique_periods = [] 
        for period in periods:
            if (period not in unique_periods) and (period not in already_processed):
                unique_periods.append(period)

        logger.info(f"Unique periods: {unique_periods}")

        other_periods = self.filter_by_log_intervals(unique_periods)#[:2]
        logger.info(f"log filter periods: {other_periods}")
        return other_periods
    
    def compute_decomposition(self, dataframe, column, detrend='linear', debug=True, steps=4): 
        p = self.get_period(dataframe, column, detrend, already_processed=[])
        if (p is None) or (len(p) == 0):
            return dataframe
        period = p[0]

        already_processed = []  
        logger = logging.getLogger(__name__)
        for i in range(1, steps+1):
            if debug:
                logger.info(f'found top period of {period // 2} minutes.')
                
            stl = STL(dataframe[column], period=period, robust=True)
            already_processed.append(period)

            res_stl = stl.fit()
            trend_stl, seasonal_stl, residual_stl = res_stl.trend, res_stl.seasonal, res_stl.resid

            dataframe['trend'] = trend_stl
            dataframe[f'seasonal_{i}'] = seasonal_stl
            dataframe['residual'] = residual_stl
            dataframe['unresolved'] = dataframe['trend'] + dataframe['residual']

            column = 'unresolved'
            p = self.get_period(dataframe, column, detrend, already_processed=already_processed)
            if (p is None) or (len(p) == 0) or (p[0] == period):
                logger.info(f'No new period found. Breaking.')
                return dataframe
            period = p[0]

        return dataframe
    
    def calculate_slope_noncentered(self, chunk, window_size, cadence, day_interval):
        logger = logging.getLogger(__name__)
        print(f"range: {(window_size), len(chunk)}")   

        for i in range(window_size, len(chunk)):
            window_data = chunk.iloc[i-window_size : i].copy()
            x = window_data.reset_index().index.values * (cadence / day_interval)
            y = window_data['unresolved'].values
            slope, intercept = np.polyfit(x, y, 1)
            
            fit_line = slope * x + intercept
            residuals = y - fit_line
            std_residual = np.std(residuals, ddof=2)

            x_mean = np.mean(x)
            spread_x = np.sum((x - x_mean)**2)

            se_slope = std_residual / np.sqrt(spread_x)

            chunk.at[chunk.index[i], 'res_std'] = std_residual
            chunk.at[chunk.index[i], 'slope'] = slope
            chunk.at[chunk.index[i], 'intercept'] = intercept
            chunk.at[chunk.index[i], 'se_slope'] = se_slope
            chunk.at[chunk.index[i], 'res'] = y[(window_size//2)] - (slope * x[(window_size//2)] + intercept)
            
        # remove the first window_size rows as they are not filled with slope values, and will be filled by previous chunk
        chunk = chunk.iloc[window_size:].copy()
        logger.info("First and last date in chunk: ", chunk.index[0], chunk.index[-1])
        print("First and last date in chunk: ", chunk.index[0], chunk.index[-1])
        return chunk

    def calculate_slope(self, chunk, window_size, cadence, day_interval):
        logger = logging.getLogger(__name__)
        print(f"range: {(window_size//2), len(chunk) - (window_size//2)}")   

        for i in range((window_size//2), len(chunk) - (window_size//2)):
            window_data = chunk.iloc[i - (window_size//2) : i + (window_size//2)].copy()
            x = window_data.reset_index().index.values * (cadence / day_interval)
            y = window_data['unresolved'].values
            slope, intercept = np.polyfit(x, y, 1)
            
            fit_line = slope * x + intercept
            residuals = y - fit_line
            std_residual = np.std(residuals, ddof=2)

            x_mean = np.mean(x)
            spread_x = np.sum((x - x_mean)**2)

            se_slope = std_residual / np.sqrt(spread_x)

            chunk.at[chunk.index[i], 'res_std'] = std_residual
            chunk.at[chunk.index[i], 'slope'] = slope
            chunk.at[chunk.index[i], 'intercept'] = intercept
            chunk.at[chunk.index[i], 'se_slope'] = se_slope
            chunk.at[chunk.index[i], 'res'] = y[(window_size//2)] - (slope * x[(window_size//2)] + intercept)
            
        # remove the first and last window_size rows as they are not filled with slope values, and will be filled by previous chunk
        chunk = chunk.iloc[(window_size//2):-(window_size//2)].copy()
        logger.info("First and last date in chunk: ", chunk.index[0], chunk.index[-1])
        print("First and last date in chunk: ", chunk.index[0], chunk.index[-1])
        return chunk

    def parallel_slope_calculation(self, dataframe, window_size, cadence, day_interval, num_processes=8, centered=True):
        logger = logging.getLogger(__name__)
        chunk_size = len(dataframe) // num_processes 
        logger.info(f"Chunk size: {chunk_size}")

        chunks = [dataframe[i:i + chunk_size].copy() for i in range(0, len(dataframe), chunk_size - window_size)]

        # if some chunks have the same end time, it means they are overlapping
        # remove duplicated end times, keep first
        # chunks = [chunk for i, chunk in enumerate(chunks) if i == 0 or chunk.index[-1] != chunks[i-1].index[-1]]

        if len(chunks) > 1:
            logger.info(f"Parallel slope calculation with window size {window_size}. Chunks: ")
            [logger.info(f'{(chunk.index[0], chunk.index[-1])}\n') for chunk in chunks]

            print(f"Parallel slope calculation with window size {window_size}. Chunks: ")
            [print(f'{(chunk.index[0], chunk.index[-1])}\n') for chunk in chunks]

            if len(chunks[-1]) < chunk_size:
                last_chunk = pd.concat([chunks[-2], chunks[-1]])
                last_chunk = last_chunk[~last_chunk.index.duplicated(keep='first')]
                chunks = chunks[:-2] + [last_chunk]

                print(f"Last chunk updated to: {(chunks[-1].index[0], chunks[-1].index[-1])}")

            with mp.Pool(processes=num_processes) as pool:
                processed_chunks = pool.starmap(self.calculate_slope if centered else self.calculate_slope_noncentered, [(chunk, window_size, cadence, day_interval) for chunk in chunks])
            final_df = pd.concat(processed_chunks)

            # print if we have duplicate indices
            print(f"Duplicate indices: {final_df.index.duplicated().sum()}")

            final_df.sort_index(inplace=True)
            return final_df
        else:
            return self.calculate_slope(dataframe, window_size, cadence, day_interval)

    def add_od_features(self, dataframe, centered=True):
        logger = logging.getLogger(__name__)
        window_size = math.ceil(24 * 60 * 2) // 24  # hourly
        dataframe.loc[::window_size + 1, 'hourly_mean_T+R'] = dataframe['unresolved'].rolling(window=window_size,
                                                                                                step=window_size + 1).mean()
        dataframe.loc[::window_size * 24 + 1, 'daily_max_T+R'] = dataframe['unresolved'].rolling(window=window_size * 24,
                                                                                                    step=window_size * 24 + 1).max()

        dataframe['hourly_mean_T+R'] = dataframe['hourly_mean_T+R'].interpolate(limit_direction='both')
        dataframe['daily_max_T+R'] = dataframe['daily_max_T+R'].interpolate(limit_direction='both')
        dataframe['orbital_decay_v0'] = dataframe['daily_max_T+R'] - dataframe['hourly_mean_T+R']

        shifted_trends = [4, 10, 15, 24, 48]  # hours
        for shift in shifted_trends:
            shifted_col_name = f'shifted_trend_{shift}h'
            dataframe[shifted_col_name] = dataframe['unresolved'].shift(-((shift * 60 * 60) // 30)).ffill()

        cadence = 30
        hour_interval = 60 * 60
        hour_count = 24
        day_count = 1
        day_interval = (day_count * hour_count * hour_interval)
        window_size = day_interval // cadence

        logger.info("Last date in df before slope calculation: ", dataframe.index[-1])

        dataframe = self.parallel_slope_calculation(dataframe, window_size, cadence, day_interval, centered=centered)

        dataframe = dataframe[~dataframe.index.duplicated(keep='first')]
        dataframe.sort_index(inplace=True)

        dataframe['orbital_decay'] = - dataframe['slope'].interpolate().ffill().bfill()
        dataframe['res'] = dataframe['res'].interpolate().ffill().bfill()
        dataframe['res_std'] = dataframe['res_std'].interpolate().ffill().bfill()
        dataframe['intercept'] = dataframe['intercept'].interpolate().ffill().bfill()
        dataframe['se_orbital_decay'] = dataframe['se_slope'].interpolate().ffill().bfill()
        dataframe['mean_altitude'] = (dataframe['trend']/1000) - 6378.137

        shifted_ods = [4, 10, 15, 24, 48]  # hours
        for shift in shifted_ods:
            shifted_col_name = f'shifted_od_{shift}h'
            dataframe[shifted_col_name] = dataframe['orbital_decay'].shift(-((shift * 60 * 60) // 30)).ffill()

        return dataframe
   
    
    def process_pod_data_nonparallel(self, lst_path, satellite_name, cadence='30s'):
        relevant_columns = ['MJD', 'a [m]', 'beta_sun [deg]']

        files_for_satellite = os.listdir(lst_path)
        files_for_satellite = [
            file for file in files_for_satellite if satellite_name in file]
        files_for_satellite.sort()

        df_sat = pd.DataFrame()
        for file in files_for_satellite:
            f = open(os.path.join(lst_path, file), 'r')
            while f.readline() != '\n':
                pass
            df_t = pd.read_fwf(f, colspecs='infer')
            columns_to_drop = [
                col for col in df_t.columns if col not in relevant_columns]
            df_t = df_t.drop(columns_to_drop, axis=1)
            df_sat = pd.concat([df_sat, df_t])

        df_sat['MJD'] = df_sat['MJD'].apply(
            lambda mjd: Time(mjd, format='mjd').iso)
        df_sat['MJD'] = pd.to_datetime(df_sat['MJD'])
        df_sat.rename({'MJD': 'time'}, axis=1, inplace=True)
        df_sat = df_sat.set_index('time')
        df_sat.index = df_sat.index.map(lambda dt: dt.round(cadence))
        return df_sat
    
    def process_pod_data(self, lst_path, satellite_name, cadence='30s'):
        files_for_satellite = [
            file for file in os.listdir(lst_path) if satellite_name in file
        ]
        files_for_satellite.sort()
        
        file_paths = [os.path.join(lst_path, file) for file in files_for_satellite]
        
        with ProcessPoolExecutor() as executor:
            dataframes = list(executor.map(process_file, file_paths))
        
        df_sat = pd.concat(dataframes, ignore_index=True)
        
        df_sat['MJD'] = Time(df_sat['MJD'].values, format='mjd').to_datetime()
        
        df_sat.rename({'MJD': 'time'}, axis=1, inplace=True)

        df_sat.sort_values(by='time', inplace=True)

        # interpolate during missing days
        df_sat.set_index('time', inplace=True)
        df_sat = df_sat.resample(cadence).asfreq()
        df_sat = df_sat.interpolate()

        df_sat.index = df_sat.index.round(cadence)

        return df_sat

    def process_pod_data_parallel_v2(self, lst_path, satellite_name, cadence='30s'):
        
        files_for_satellite = [
            file for file in os.listdir(lst_path) if satellite_name in file
        ]
        files_for_satellite.sort()
        
        file_paths = [os.path.join(lst_path, file) for file in files_for_satellite]
        
        with ProcessPoolExecutor() as executor:
            dataframes = list(executor.map(process_file, file_paths))
        
        df_sat = pd.concat(dataframes, ignore_index=True)

        df_sat['MJD'] = Time(df_sat['MJD'].values, format='mjd').to_datetime()
        df_sat.rename({'MJD': 'time'}, axis=1, inplace=True)
        df_sat.sort_values(by='time', inplace=True)
        df_sat.set_index('time', inplace=True)
        df_sat.index = df_sat.index.map(lambda dt: dt.round(cadence))

        # interpolate during missing days
        df_sat = df_sat.interpolate()

        return df_sat

class TLEPreprocessor():
    def preprocess_tles(self, tle_path, out_path, cadence='30s', tle_config_path=None):
        for file in os.listdir(tle_path):
            if file.endswith(".csv"):
                df = self.process_tle_data(os.path.join(tle_path, file), cadence=cadence)
                df.to_parquet(f"{out_path}/{file}", index=True, index_label='time')

    def process_tle_data(self, tle_sat_path, cadence='30s'):
        relevant_columns = ['EPOCH', 'SEMIMAJOR_AXIS']

        df_sat = pd.read_csv(tle_sat_path)
        df_sat = df_sat[relevant_columns].copy()

        df_sat['EPOCH'] = pd.to_datetime(df_sat['EPOCH'])
        df_sat.rename(
            {'EPOCH': 'time', 'SEMIMAJOR_AXIS': 'a [m] TLE'}, axis=1, inplace=True)

        df_sat = df_sat.set_index('time')
        df_sat.index = df_sat.index.map(lambda dt: dt.round(cadence))

        df_sat = df_sat.resample(cadence).mean()
        df_sat = df_sat.interpolate()

        start_time = df_sat.index[0]
        end_time = df_sat.index[-1]

        start_time = start_time.replace(hour=0, minute=0, second=0)
        end_time = end_time.replace(hour=23, minute=59, second=30)

        padding = pd.DataFrame(index=pd.date_range(
            start=start_time, end=df_sat.index[0], freq=cadence))
        padding['a [m] TLE'] = np.nan
        df_sat = pd.concat([padding, df_sat])

        padding = pd.DataFrame(index=pd.date_range(
            start=df_sat.index[-1], end=end_time, freq=cadence))
        padding['a [m] TLE'] = np.nan
        df_sat = pd.concat([df_sat, padding])

        df_sat.ffill(inplace=True)
        df_sat.bfill(inplace=True)

        df_sat.sort_index(inplace=True)

        return df_sat


def process_smos_file(file_path):
    try:
        column_names = [
            "date", "time", "SunBT_1AU[K]", "SunFlux_1AU[sfu]", "N_Snapshots", 
            "SunBT_1AU_STD[K]", "Mean_Sun_Elevation[rad]", "Earth-Sun_Distance_Factor", 
            "SMOS_Orbit_Number", "SMOS_Orbit_ANX_Start_Date", "SMOS_Orbit_ANX_Start_Time",
            "Acquisition_Start_Date", "Acquisition_Start_Time",
            "Acquisition_Stop_Date", "Acquisition_Stop_Time"
        ]
        df = pd.read_csv(
            file_path, skiprows=2, header=None, delim_whitespace=True, names=column_names
        )
        if df.empty:
            return None
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df.set_index('timestamp', inplace=True)
        df_flux = df[['SunFlux_1AU[sfu]']].copy()
        df_flux.rename(columns={'SunFlux_1AU[sfu]': 'sun_flux_smos'}, inplace=True)
        return df_flux
    except Exception as e:
        logging.error(f"Failed to parse SMOS file {file_path}: {e}")
        return None

class SMOSPreprocessor:
    def preprocess_smos(self, input_path: str, output_path: str, cadence='30s', interpolate=False):
        smos_path = Path(input_path)
        if not smos_path.exists():
            logging.warning(f"SMOS data directory not found at: {smos_path}")
            return

        smos_files = sorted(list(smos_path.glob('**/*.txt')))
        if not smos_files:
            logging.warning("No SMOS .txt files found.")
            return

        logging.info(f"Found {len(smos_files)} SMOS files to process.")
        
        with ProcessPoolExecutor() as executor:
            dataframes = list(executor.map(process_smos_file, smos_files))
        
        df_final = pd.concat([df for df in dataframes if df is not None])

        if df_final.empty:
            logging.warning("SMOS data is empty after processing all files.")
            return

        df_final.sort_index(inplace=True)
        df_final = df_final.resample(cadence).mean()
        if interpolate:
            df_final = df_final.interpolate()
            
        df_final.to_parquet(output_path)
        logging.info(f"Successfully loaded and processed SMOS data to {output_path}")


class GFZPreprocessor:
    def preprocess_gfz(self, input_path: str, output_path: str, cadence='30s', interpolate=False):
        df = pd.read_csv(input_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.resample(cadence).asfreq()
        if interpolate:
            df = df.interpolate()
        df.to_parquet(output_path)
        logging.info(f"Successfully loaded and processed GFZ data to {output_path}")


class DataImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean', columns=None, all_cols=False, fill_value=None):
        self.strategy = strategy
        self.columns = columns
        self.all_cols = all_cols
        self.fill_value = fill_value

        if strategy == 'knn':
            self.imputer = KNNImputer()
            return

        if strategy in ['mean', 'median', 'most_frequent']:
            self.imputer = SimpleImputer(strategy=strategy)
            return

        if strategy in ['interpolate', 'forward']:
            self.imputer = None
            return

        if strategy == 'constant':
            self.imputer = SimpleImputer(
                strategy=strategy, fill_value=fill_value)
            return

        raise Exception(f"unimplemented numerical strategy: {strategy}")

    def fit(self, X, y=None):
        columns_with_na = X.columns[X.isna().any()].tolist()
        columns_with_na = [col for col in columns_with_na if col != '']

        cols_to_impute = self.columns if not self.all_cols else columns_with_na

        if len(cols_to_impute) > 0:
            if self.imputer != None:
                self.imputer.fit(X[cols_to_impute])
            # else nothing to fit

        return self

    def transform(self, X, y=None):
        logger = logging.getLogger(__name__)
        columns_with_na = X.columns[X.isna().any()].tolist()
        columns_with_na = [col for col in columns_with_na if col != '']

        cols_to_impute = self.columns if not self.all_cols else columns_with_na

        logger.info(
            f"Imputing columns using strategy {self.strategy}. Columns:\n {cols_to_impute}")

        if len(cols_to_impute) > 0:
            if self.imputer != None:
                X[cols_to_impute] = self.imputer.transform(X[cols_to_impute])
            else:
                if self.strategy == 'interpolate':
                    X[cols_to_impute] = X[cols_to_impute].interpolate(
                        method='linear', limit_direction="both")
                elif self.strategy == 'forward':
                    X[cols_to_impute] = X[cols_to_impute].fillna(
                        method='ffill')
                    # backwards as well in case first N values are None
                    X[cols_to_impute] = X[cols_to_impute].fillna(
                        method='bfill')

        return X

    def fit_transform(self, X, y=None):
        r = self.fit(X).transform(X)
        return r


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None, columns_to_keep=None):
        self.columns_to_drop = columns_to_drop
        self.columns_to_keep = columns_to_keep

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cols_to_drop = [
            col for col in self.columns_to_drop if col in list(X.columns)]
        cols_to_drop += [col for col in list(X.columns)
                         if col not in self.columns_to_keep]

        X = X.drop(cols_to_drop, axis=1)
        X = X.dropna(axis=1, how='all')
        X = X.loc[:, ~X.columns.duplicated()]
        return X

    def fit_transform(self, X, y=None):
        r = self.fit(X).transform(X)
        return r


class ColumnRenamer(BaseEstimator, TransformerMixin):
    def __init__(self, rename_dict=None, reorder=True):
        self.rename_dict = rename_dict
        self.reorder = reorder

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.rename(columns=self.rename_dict)

        if self.reorder:
            X = X.reindex(self.rename_dict.values(), axis=1)
        return X

    def fit_transform(self, X, y=None):
        r = self.fit(X).transform(X)
        return r


class DataResampler(BaseEstimator, TransformerMixin):
    def __init__(self, cadence="1min"):
        self.cadence = cadence

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.resample(self.cadence).mean()

    def fit_transform(self, X, y=None):
        r = self.fit(X).transform(X)
        return r


class TimeIndexSetter(BaseEstimator, TransformerMixin):
    def __init__(self, column=None):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # reorder columns
        X[self.column] = pd.to_datetime(X[self.column])
        return X.set_index(self.column)

    def fit_transform(self, X, y=None):
        r = self.fit(X).transform(X)
        return r


class CustomProcessor(BaseEstimator, TransformerMixin):
    def replace_max_with_nan(self, col):
        max_val = col.max()

        patterns = [
            r'9+(\.(0|(9+)))?$',
            r'0\.9+$'
        ]

        if any(re.match(pattern, str(max_val)) for pattern in patterns):
            col[col == max_val] = np.nan

        return col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # reorder columns
        X.apply(self.replace_max_with_nan, axis=0)
        return X

    def fit_transform(self, X, y=None):
        r = self.fit(X).transform(X)
        return r


class PipelinesProcessor:
    def __init__(self, pipeline_configuration, dataset_path):
        self.pipeline_configuration = pipeline_configuration
        self.dataset_path = dataset_path

    def preprocess(self):
        pipe = []
        # 0. replace NaN
        custom_preprocessing = self.pipeline_configuration.get('custom')

        if custom_preprocessing and custom_preprocessing.get('replace_max_with_nan'):
            pipe.append(
                ("Reintroduce NaN instead of 9.99.. max.", CustomProcessor()))

        with open(f"{self.dataset_path.replace('.csv', '').replace('raw','configs')}/config.json", 'r') as file:
            column_dict = json.load(file)

        # 1. drop columns not in variable list or are marked to drop
        pipe.append(("Drop columns.", ColumnDropper(columns_to_drop=self.pipeline_configuration['columns_to_drop'], columns_to_keep=list(map(
            lambda v: v['Name'],  column_dict)))))

        # 2. rename columns
        if self.pipeline_configuration['rename_columns']:
            rename_dict = {v['Name']: v['NewName'] for v in column_dict}
            pipe.append(("Rename columns.", ColumnRenamer(rename_dict=rename_dict,
                        reorder=self.pipeline_configuration['reorder_columns'])))

        # 3. set index
        pipe.append(("Set time index.", TimeIndexSetter(
            column=self.pipeline_configuration['index_column'])))

        # 4. resample data to cadence
        if self.pipeline_configuration.get('set_cadence'):
            pipe.append(("Resample data.", DataResampler(
                cadence=self.pipeline_configuration['set_cadence'])))

        # 5. input specifically mentioned columns
        for scheme in self.pipeline_configuration['imputing_schemes']['specific']:
            columns = self.pipeline_configuration['imputing_schemes']['specific'][scheme]['columns']
            if len(columns) > 0:
                fill_value = self.pipeline_configuration['imputing_schemes']['specific'][scheme].get(
                    'value', None)
                pipe.append((f"Imput missing values (specific columns) with scheme {scheme}.",
                             DataImputer(strategy=scheme,
                                         columns=columns,
                                         fill_value=fill_value)))

        # 6. input rest columns
        pipe.append(
            (f"Imput missing values (other columns) with scheme {self.pipeline_configuration['imputing_schemes']['others_scheme']}.",
             DataImputer(strategy=self.pipeline_configuration['imputing_schemes']['others_scheme'],
                         all_cols=True)))

        return Pipeline(pipe)


def parse_config(ctx, param, value):
    if value is None:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise click.BadParameter('Invalid JSON format')
