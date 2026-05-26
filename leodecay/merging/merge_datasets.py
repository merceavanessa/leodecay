import click
import logging
import pandas as pd
import datetime
import os
import json

from scipy.signal import savgol_filter


@click.command()
@click.option('-input_file_paths', type=click.Path(exists=True), multiple=True,
              help='List of input file paths to merge')
@click.option('-satellite_base_path', type=click.Path(exists=True), help='Satellite data base path')
@click.option('-tle_base_path', type=click.Path(exists=True), help='TLE data base path')
@click.option('-output_base_path', type=click.Path(exists=False), help='Output base path')
@click.option('-cadence', type=str, default='30s', help='Resampling cadence, e.g., 30s, 1min')
def main(input_file_paths, satellite_base_path, tle_base_path, output_base_path, cadence='30s'):
    logging.info(
        f"Merging datasets: {input_file_paths}, {satellite_base_path}, {tle_base_path}\n")

    # merge solar and geomagnetic data
    merged_df = pd.read_csv(input_file_paths[0])
    merged_df['time'] = pd.to_datetime(merged_df['time'], utc=True)

    for file in input_file_paths[1:]:
        df = pd.read_csv(file)
        df['time'] = pd.to_datetime(df['time'], utc=True)

        merged_df = merged_df.merge(df, on='time', how='left')
        logging.info(f"Merged {file} with shape {df.shape}, resulting shape: {merged_df.shape}. Duplicates in merged_df: {merged_df['time'].duplicated().sum()}. Is monotonic increasing: {merged_df['time'].is_monotonic_increasing}\n")

    merged_df = merged_df.dropna(axis=1, how='all')
    merged_df = merged_df.drop_duplicates()

    merged_df['time'] = pd.to_datetime(merged_df['time'], utc=True)
    merged_df = merged_df.set_index('time')
    merged_df = merged_df.resample(cadence).mean()

    # for each satellite, merge POD with the other datasets
    satellite_dict = json.load(open(f"{satellite_base_path.replace('preprocessed', 'configs')}/config.json"))
    for key in satellite_dict:
        if (os.path.isfile(os.path.join(satellite_base_path, f'{key}.csv'))):
            df_sat = pd.read_csv(os.path.join(satellite_base_path, f'{key}.csv'))

            # ToDo - remove this once the RK data has correctly the index name saved (bug fixed but dont' want to regenerate all the data yet)
            if 'Unnamed: 0' in df_sat.columns:
                df_sat = df_sat.rename(columns={'Unnamed: 0': 'time'})

            df_sat['time'] = pd.to_datetime(df_sat['time'], utc=True)
            df_sat = df_sat.set_index('time')

            if 'orbital_decay' not in df_sat.columns:
                aDot_smooth_col = df_sat.columns[df_sat.columns.str.contains('aDot_smooth')]
                df_sat['orbital_decay'] = - df_sat[aDot_smooth_col]

            df_tle = pd.read_csv(os.path.join(tle_base_path, f'{key}.csv'))
            # for TLEs data, compute decay rates using a savgol filter (equivalent to 24h rolling window + gradient)
            df_tle['orbital_decay_tle'] = - savgol_filter(df_tle['a [m] TLE'], window_length=24 * 60 * 2 - 1,
                                                          polyorder=1, deriv=1) * 2880
            df_tle['time'] = pd.to_datetime(df_tle['time'], utc=True)
            df_tle = df_tle.set_index('time')

            df_sat = df_sat.merge(df_tle, left_index=True, right_index=True)
            df_sat = df_sat.merge(merged_df, left_index=True, right_index=True)

            df_sat.to_parquet(os.path.join(output_base_path, f'{key}.parquet'), index=True)

    logging.info(f"Successfully merged.\n")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = f'./logs/make_merge-{datetime.datetime.now()}.log'

    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        filename=log_file,
        filemode='a'
    )

    main()
