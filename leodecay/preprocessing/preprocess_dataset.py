import pprint
from ..preprocessing.preprocessing_utils import *
import yaml
import datetime
import logging
import logging.handlers
from ..utils.logging_utils import setup_logging

def parse_config(ctx, param, value):
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        raise click.BadParameter('Invalid YAML format')

@click.command()
@click.option('-input_file_path', type=click.Path(exists=True), help='Path to the input file')
@click.option('-dataset', type=str, help='Dataset name')
@click.option('-pipeline_configuration', callback=parse_config, help='Configuration as a YAML string')
@click.option('-output_file_path', type=click.Path(exists=False), help='Path to the output file')
def main(input_file_path, dataset, pipeline_configuration, output_file_path):
    pp = pprint.PrettyPrinter(indent=2, sort_dicts=False)
    if dataset == "TLE":
        processor = TLEPreprocessor()
        processor.preprocess_tles(input_file_path, output_file_path, pipeline_configuration.get('set_cadence', '30s'))
        logging.info(f"TLE data processed successfully to remove unused columns. {input_file_path} -> {output_file_path}, {os.listdir(input_file_path)}\n")
    elif dataset == "POD":
        processor = PODPreprocessor()
        processor.preprocess_pod(input_file_path, output_file_path, pipeline_configuration.get('set_cadence', '30s'))
        logging.info(f"POD data processed successfully from LST files.\n")
    elif dataset == "SMOS":
        processor = SMOSPreprocessor()
        processor.preprocess_smos(input_file_path, output_file_path, pipeline_configuration.get('set_cadence', '30s'))
        logging.info(f"SMOS data processed successfully. {input_file_path} -> {output_file_path}\n")
    elif dataset == "GFZ":
        processor = GFZPreprocessor()
        processor.preprocess_gfz(input_file_path, output_file_path, pipeline_configuration.get('set_cadence', '30s'))
        logging.info(f"GFZ data processed successfully. {input_file_path} -> {output_file_path}\n")
    else:
        pipeline = PipelinesProcessor(pipeline_configuration, input_file_path).preprocess()
        df = pd.read_csv(f"{input_file_path}")
        df = pipeline.fit_transform(df)

        if dataset != 'LATIS':
            processor = OMNIPreprocessor()
            df = processor.calculate_position_anomalies(df, id_type="IMF")
            df[(df['Spacecraft ID (Plasma)'] != -1)] = processor.calculate_position_anomalies(df[(df['Spacecraft ID (Plasma)'] != -1)].copy(), id_type="Plasma")
            df = processor.add_additional_features(df)
            df.sort_index(inplace=True)
            df = processor.add_slopes(df, pipeline_configuration['columns_to_calculate_slopes'])

        assert not df.index.duplicated().any()

        df.sort_index(inplace=True)
        df.to_csv(f"{output_file_path}")
        logging.info(f"Pipeline applied successfully.\n Resulting pipeline: { pp.pformat(pipeline)}")

if __name__ == '__main__':
    log_file = f'./logs/make_preprocessing-{datetime.datetime.now()}.log'  
    listener = setup_logging(log_file)

    try:
        main()
    finally:
        listener.stop()
