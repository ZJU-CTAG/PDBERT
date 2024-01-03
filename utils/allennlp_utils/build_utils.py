from typing import Optional

from allennlp.common import Params
from allennlp.data import DatasetReader


def build_dataset_reader_from_config(config_path,

                                     serialization_dir: Optional[str] = None) -> DatasetReader:
    """
    This util method help build datset reader from params in jsonnet file.
    Both jsonnet and json files are acceptable.
    :param serialization_dir: Parameter of 'DatasetReader.from_params'.
    """
    dataset_reader_params = Params.from_file(config_path)['dataset_reader']
    # build dataset reader from params
    dataset_reader = DatasetReader.from_params(dataset_reader_params,
                                               serialization_dir=serialization_dir)
    return dataset_reader


def build_dataset_reader_from_dict(config_dict,
                                   serialization_dir: Optional[str] = None) -> DatasetReader:
    """
    This util method help build datset reader from dict-type config.
    """
    dataset_reader_params = Params(config_dict)
    # build dataset reader from params
    dataset_reader = DatasetReader.from_params(dataset_reader_params,
                                               serialization_dir=serialization_dir)
    return dataset_reader
