import yaml
import logging


log_file='logs/general_logs/training.log'
log_format='%(levelname)s: %(asctime)s: %(message)s'
logging.basicConfig(filename=log_file,level=logging.INFO,format=log_format)

def read_config(config_path):
    logging.info('>>>>> Extracting config.yaml')
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)
    return content
    