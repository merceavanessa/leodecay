# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from src.data.cdaweb_downloader import CdaWebLoader
from src.data.lasp_downloader import LatisLoader
from src.data.tle_downloader import TLELoader
from dotenv import find_dotenv, load_dotenv
from enum import Enum


class EnumType(click.Choice):
    def __init__(self, enum, case_sensitive=False):
        self.__enum = enum
        super().__init__(
            choices=[item.value for item in enum], case_sensitive=case_sensitive)

    def convert(self, value, param, ctx):
        converted_str = super().convert(value, param, ctx)
        return self.__enum(converted_str)


class DataSource(Enum):
    OMNI = "OMNI"
    LATIS = "LATIS"
    TLE = "TLE"


@click.command()
@click.option('--folder_path', type=click.STRING)
@click.option('--conf_path', type=click.STRING)
@click.option('--data_source', type=EnumType(DataSource))
@click.option('--dataset', type=click.STRING)
@click.option('--start_date', type=click.STRING)
@click.option('--end_date', type=click.STRING)
@click.option('--username', type=click.STRING, default=None)
@click.option('--password', type=click.STRING, default=None)
def main(folder_path, conf_path=None, data_source=None, dataset=None, start_date="1995-01-01", end_date="2024-03-03", username=None, password=None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    if data_source == DataSource.OMNI:
        loader = CdaWebLoader(folder_path=folder_path, dataset=dataset, start_date=start_date, end_date=end_date)
        logging.info(f"Combined intermediate data saved to {loader.load_data()}")

    elif data_source == DataSource.LATIS:
        loader = LatisLoader(folder_path=folder_path, dataset=dataset, start_date=start_date, end_date=end_date)
        logging.info(f"Combined intermediate data saved to {loader.load_data()}")
    elif data_source == DataSource.TLE:
        loader = TLELoader(folder_path=folder_path, conf_path=conf_path, dataset=dataset, start_date=start_date, end_date=end_date, username=username, password=password)
        logging.info(f"Combined intermediate data saved to {loader.load_data()}")
    else:
        logging.warning("Please provide the data source")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # , filename='make_dataset.log')
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
