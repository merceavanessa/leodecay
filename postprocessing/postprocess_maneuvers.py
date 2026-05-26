import click
import os
import logging
from pathlib import Path
import pandas as pd
from dotenv import find_dotenv, load_dotenv
import datetime
import logging
import logging.handlers
import multiprocessing
import tqdm
import numpy as np
import json
from preprocessing.preprocessing_utils import PODPreprocessor
from utils.logging_utils import setup_logging

# not relevant for the two-year analysis, where this process is dealth with in the decay derivation

def get_intervals_to_reprocess(df):
    df['is_man_or_missing'] = df['is_maneuver_day'] | df['is_missing_day']
    column = 'is_man_or_missing'
    missing_day_starts = df[df[column] & ~
                            df[column].shift(1).fillna(False)].index
    missing_day_ends = df[df[column] & ~df[column].shift(
        -1).fillna(False)].index + pd.Timedelta(30, unit='s')

    intervals = []
    for i in range(len(missing_day_starts)):
        if i == 0:
            start = df.index[0]
            end = missing_day_starts[i]
        elif i == len(missing_day_starts)-1:
            start = missing_day_ends[i]
            end = df.index[-1]
        else:
            start = missing_day_ends[i-1]
            end = missing_day_starts[i]

        if (end - start).days > 14:
            end_new = start + pd.Timedelta(days=14)
            intervals.append((start, end_new))

            start_new = end - pd.Timedelta(days=14)
            intervals.append((start_new, end))
        else:
            intervals.append((start, end))

    return intervals


def update_flags(df):
    df['is_maneuver_unresolved'] = False
    df['is_maneuver_unresolved_10m_decay'] = False
    df.loc[(df['orbital_decay'] < 0) & (~df['is_maneuver_day_extended']), [
        'is_maneuver_unresolved']] = True
    df.loc[(df['orbital_decay'] < -10) & (~df['is_maneuver_day_extended']),
           ['is_maneuver_unresolved_10m_decay']] = True

    df['is_maneuver_period_generic'] = False
    df['is_maneuver_period_generic'] = df['is_maneuver_day_extended'] | df['is_missing_day_extended']

    if 'res' not in df.columns:
        rescol = 'Residuals (m)'
    else:
        rescol = 'res'

    df[['orbital_decay_c', f"{rescol}_c", 'res_std_c', 'unresolved_c', 'mean_altitude_c', 'trend_c', 'residual_c']] = df[['orbital_decay', rescol, 'res_std', 'unresolved', 'mean_altitude', 'trend', 'residual']]
        
    df.loc[df['is_maneuver_period_generic'], ['orbital_decay_c', 'res_std_c', f"{rescol}_c",
                                              'unresolved_c', 'mean_altitude_c', 'trend_c', 'residual_c']] = np.nan
    df.loc[df['is_maneuver_period_generic'], ['seasonal_1_c',
                                              'seasonal_2_c', 'seasonal_3_c', 'seasonal_4_c']] = np.nan

    return df


def add_shifts(df, how='hours'):
    if how == 'hours':
        shifted_ods = list(range(1, 24))
        for shift in shifted_ods:
            shifted_col_name = f'shifted_od_{shift}h'
            df[shifted_col_name] = df['orbital_decay_c'].shift(
                -((shift * 60 * 60) // 30)).ffill()
    elif how == 'minutes':
        shifted_ods = list(range(1, 24 * 60, 30))
        for shift in shifted_ods:
            shifted_col_name = f'shifted_od_{shift}m'
            df[shifted_col_name] = df['orbital_decay_c'].shift(
                -((shift * 60) // 30)).ffill()
    else:
        raise ValueError('how must be either "hours" or "minutes"')

    return df


def add_missing_day_entries(missing_days, df):
    for missing_day_Y_j in missing_days:
        if missing_day_Y_j in df.index.strftime('%Y-%j'):
            print(f'{missing_day_Y_j} already in df')
            continue
        missing_day_datetime = datetime.datetime.strptime(
            missing_day_Y_j, '%Y-%j')
        missing_day = pd.date_range(
            missing_day_datetime, periods=24*60*2, freq='30s')
        missing_day_df = pd.DataFrame(index=missing_day)
        missing_day_df['is_maneuver_day'] = False
        df = pd.concat([df, missing_day_df])

    df.sort_index(inplace=True)
    return df


def expand_flag(df, flag='is_maneuver_day'):
    true_indices = df.index[df[flag] == True]
    extended_true_indices = pd.DatetimeIndex([])

    for date in true_indices.normalize().unique():
        start_date = date - pd.Timedelta(hours=12)
        end_date = date + pd.Timedelta(days=1, hours=12)
        extended_true_indices = extended_true_indices.union(
            df[start_date:end_date].index
        )

    df[f'{flag}_extended'] = False
    df.loc[extended_true_indices, f'{flag}_extended'] = True
    return df


@click.command()
@click.option('-input_file_path', type=click.Path(exists=True), help='Path to the input file')
@click.option('-maneuver_file_path', type=click.Path(exists=True), help='Path to the maneuvers file')
@click.option('-output_file_path', type=click.Path(exists=False), help='Path to the output file')
def main(input_file_path, maneuver_file_path, output_file_path):
    cfg = json.load(
        open(f"{input_file_path.replace('processed', 'configs/POD')}/config.json"))
    satellite_names = cfg.keys()
    logger = logging.getLogger(__name__)
    
    for satellite_name in satellite_names:
        logger.info(
            f"Postprocessing {satellite_name}... Started at time {datetime.datetime.now()}")

        df = pd.read_csv(os.path.join(
            input_file_path, f'{satellite_name}.csv'), index_col='time')
        df.index = pd.to_datetime(df.index)

        missing_days = cfg[satellite_name]['missing_days'] if 'missing_days' in cfg[satellite_name] else []
        df = add_missing_day_entries(missing_days, df)

        maneuvers = pd.read_csv(maneuver_file_path)
        # maneuvers['Maneuver Start'] = pd.to_datetime(maneuvers['Maneuver Start'])

        # if 'DoY' not in maneuvers.columns:
        #     maneuvers['DoY'] = pd.to_datetime(maneuvers['Maneuver Start']).dt.strftime('%Y-%j')
        
        # maneuvers.to_csv(maneuver_file_path)
            
        maneuvers_SAT = maneuvers[maneuvers['Satellite']
                                  == cfg[satellite_name]['name'].upper()]['DoY'].unique()
        
        print(f"Maneuvers: {maneuvers_SAT}, CFG: {cfg[satellite_name]['name']}")

        df['DoY'] = df.index.strftime('%Y-%j')
        df['is_maneuver_day'] = df.apply(
            lambda x: x['DoY'] in maneuvers_SAT, axis=1)
        df['is_missing_day'] = df.apply(
            lambda x: x['DoY'] in missing_days, axis=1)

        outlier_indices = df[df['orbital_decay']
                             < -5000].index  # 5 km threshold
        if len(outlier_indices) > 0:
            outlier_days = df.loc[outlier_indices]['DoY'].unique()
            df['is_outlier_maneuver_day'] = df.apply(
                lambda x: x['DoY'] in outlier_days, axis=1)

        df = expand_flag(df, flag='is_maneuver_day')
        df = expand_flag(df, flag='is_missing_day')
        intervals = get_intervals_to_reprocess(df)
        logger.info(f'Intervals to reprocess: {intervals}')

        if len(intervals) == 0:
            df = update_flags(df)
            df = add_shifts(df)
            logger.info(f"No intervals to reprocess for {satellite_name}")
            df.to_csv(os.path.join(output_file_path,
                      f'{satellite_name}.csv'), index=True, index_label='time')
            continue

        proc = PODPreprocessor()
        chunks = [df[(df.index >= start) & (df.index < end)]
                    [['a [m]']].copy() for start, end in intervals]
        args = [(i, chunk) for i, chunk in enumerate(chunks)]

        with multiprocessing.Pool(processes=8) as pool:
            processed_chunks = list(tqdm.tqdm(pool.imap_unordered(
                proc.process_chunk, args), total=len(args)))

        for i, chunk in enumerate(processed_chunks):
            if len(chunk) == 0:
                continue
            
            logger.info(chunk.index[0], chunk.index[-1])
            df.loc[chunk.index, 'unresolved'] = chunk['unresolved']
            df.loc[chunk.index, 'trend'] = chunk['trend']
            df.loc[chunk.index, 'residual'] = chunk['residual']

            for i in range(1, 5):
                df.loc[chunk.index,
                       f'seasonal_{i}'] = chunk[f'seasonal_{i}']

        df = proc.add_od_features(df)
        df = update_flags(df)
        df = add_shifts(df)

        assert not df.index.duplicated().any()
        assert df.index.is_monotonic_increasing

        df.to_csv(os.path.join(output_file_path,
                  f'{satellite_name}.csv'), index=True, index_label='time')

        logging.info(f"Removal applied successfully.\n")


if __name__ == '__main__':
    log_file = f'./logs/make-postprocess-maneuvers-{datetime.datetime.now()}.log'
    listener = setup_logging(log_file)

    try:
        main()
    finally:
        listener.stop()
