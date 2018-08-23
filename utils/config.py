import json
from easydict import EasyDict
import os


def get_config_from_json(json_file):
    # get the config from a json file

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.summary_dir = os.path.join('../experiments', config.exp_name, 'summary/')
    config.checkpoint_dir = os.path.join('../experiments', config.exp_name, 'checkpoint/')
    return config


if __name__ == '__main__':
    from utils import get_args
    args = get_args()
    print(args)
    config, config_dict = get_config_from_json(args.config)
    print(config)
    print(config_dict)
    config = process_config(args.config)
    print(config)
    print(type(config))
