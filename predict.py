import os
from typing import Union, List

import numpy as np
import torch
import yaml

from data_loader import DataLoader
from helper import OutputMode, ModelMode
from model import TICIModelHandler


class TICIScorer:
    """Wraps model and preprocessor, must be initialized with corresponding arguments from config. Uses three public
    functions as an interface to process series given by the corresponding paths.
    Config path is the path used load the config with the corresponding model_parameters"""

    def __init__(self, config: Union[dict, Union[str, os.PathLike]] = 'configuration.yml', ensemble=False):
        """gets the configuration file or path for preprocessing and network to build the model and data loader
        args:
            config: Pathlike object pointing to a valid yaml to initialize model and dataloader or dictionary with
                    corresponding keyword arguments
            ensemble: model makes prediction based on a set of multiple weights"""

        if isinstance(config, str):
            with open(config, 'r') as handle:
                config = yaml.load(handle, Loader=yaml.FullLoader)

        if not ensemble and isinstance(config['model_params']['pretrained'], list):
            config['model_params']['pretrained'] = config['model_params']['pretrained'][0]
        # maps output to a TICI-Score
        try:
            self.label_mapping = config['label_mapping']
        except KeyError:
            self.label_mapping = None
        # loads dicoms into ram and performs preprocessing
        self.data_loader = DataLoader(config['preprocessor_steps'])
        # DL model for TICI-Score prediction
        self.model = TICIModelHandler(**config['model_params'])

    def __call__(self, dcm_path: List[Union[str, os.PathLike]], output_mode: OutputMode = OutputMode.last_frame,
                 softmax=False) -> torch.Tensor:
        """Implementation of Dataloading and forward path of the DL-Model.
            args:
                dcm_path: dicom paths to frontal and lateral view of an DSA-Series (order: [frontal, lateral])
                output_mode: valid values: 'last_frame' -> only the output of the last frame is returned
                                            'all_frames' -> the output of all frames will be returned
                softmax: True -> the softmax values are returned; False -> argmax and label_maping is applied to the
                         output
            return:
                score: tici-score predicted by the model either softmax or score based on mapping"""

        data = self.data_loader(dcm_path)
        raw_values = self.model(*data, model_mode=ModelMode.inference, output_mode=output_mode)
        if not softmax:
            score = [float(torch.argmax(raw_value, dim=-1)) for raw_value in raw_values]
            if self.label_mapping:
                score = [self.label_mapping[temp_score] for temp_score in score]
            if len(score) == 1:
                score = score[0]
        return score

    # def __call__(self, batch, series_lengths, output_mode='all_frames'):
    #     return self.model(batch, series_lengths, mode='train', output_mode=output_mode)


def predict_series(config: Union[dict, Union[str, os.PathLike]] = 'configuration.yml',
                   dcm_path=List[Union[str, os.PathLike, List[Union[str, os.PathLike]]]],
                   output_mode: OutputMode = OutputMode.last_frame,
                   softmax=False, ensemble=False) -> Union[str, int, np.ndarray, List[Union[str, int, np.ndarray]]]:
    """wrapping function for TICIScorer init and call. Also enables to predict a list of multiple DSA-Series.
         args:
            config: Pathlike object pointing to a valid yaml to initialize model and dataloader or dictionary with
                    corresponding keyword arguments
            dcm_path: dicom paths to frontal and lateral view of an DSA-Series (order: [frontal, lateral]). Can be
                     a list of multiple series (list of lists).
            output_mode: valid values: 'last_frame' -> only the output of the last frame is returned
                                        'all_frames' -> the output of all frames will be returned
            softmax: True -> the softmax values are returned; False -> argmax and label_maping is applied to the
                     output
            ensemble: model makes prediction based on a set of multiple weights
        return:
            score: tici-score predicted by the model either softmax or score based on mapping. If input is a list
                   of multiple series a list is returned"""

    device_model = TICIScorer(config, ensemble=ensemble)
    # if dcm_path is a list of series iterate through it.
    if isinstance(dcm_path[0], list):
        score = []
        for path in dcm_path:
            score.append(device_model(path, output_mode, softmax))
    else:
        score = device_model(dcm_path, output_mode, softmax)
    return score


def predict_experiment_I(dcm_path=List[Union[str, os.PathLike, List[Union[str, os.PathLike]]]],
                         output_mode: OutputMode = OutputMode.last_frame, softmax=False) -> Union[
    str, int, np.ndarray, List[Union[str, int, np.ndarray]]]:
    """Predicts results from "Deep learning-based automated TICI scoring: a timely proof-of-principle study" -
    experiment I.
    Calls predict_series with proper initilization.
         args:
            config: Pathlike object pointing to a valid yaml to initialize model and dataloader or dictionary with
                    corresponding keyword arguments
            dcm_path: dicom paths to frontal and lateral view of an DSA-Series (order: [frontal, lateral]). Can be
                     a list of multiple series (list of lists).
            output_mode: valid values: 'last_frame' -> only the output of the last frame is returned
                                        'all_frames' -> the output of all frames will be returned
            softmax: True -> the softmax values are returned; False -> argmax and label_maping is applied to the
                     output
        return:
            score: tici-score predicted by the model either softmax or score based on mapping. If input is a list
                   of multiple series a list is returned"""
    config = 'configuration.yml'
    if isinstance(config, str):
        with open(config, 'r') as handle:
            config = yaml.load(handle, Loader=yaml.FullLoader)
    config['model_params']['pretrained'] = config['model_params']['pretrained'][0]
    return predict_series(config, dcm_path, output_mode, softmax, True)


def predict_experiment_III(dcm_path=List[Union[str, os.PathLike, List[Union[str, os.PathLike]]]],
                           output_mode: OutputMode = OutputMode.last_frame, softmax=False) -> Union[
    str, int, np.ndarray, List[Union[str, int, np.ndarray]]]:
    """Predicts results from "Deep learning-based automated TICI scoring: a timely proof-of-principle study" -
    experiment III, all three folds are ensembled.
    Calls predict_series with proper initilization.
         args:
            config: Pathlike object pointing to a valid yaml to initialize model and dataloader or dictionary with
                    corresponding keyword arguments
            dcm_path: dicom paths to frontal and lateral view of an DSA-Series (order: [frontal, lateral]). Can be
                     a list of multiple series (list of lists).
            output_mode: valid values: 'last_frame' -> only the output of the last frame is returned
                                        'all_frames' -> the output of all frames will be returned
            softmax: True -> the softmax values are returned; False -> argmax and label_maping is applied to the
                     output
        return:
            score: tici-score predicted by the model either softmax or score based on mapping. If input is a list
                   of multiple series a list is returned"""
    config = 'configuration.yml'
    if isinstance(config, str):
        with open(config, 'r') as handle:
            config = yaml.load(handle, Loader=yaml.FullLoader)
    config['model_params']['pretrained'] = config['model_params']['pretrained'][1:]
    return predict_series(config, dcm_path, output_mode, softmax, True)


def predict_ensemble(dcm_path=List[Union[str, os.PathLike, List[Union[str, os.PathLike]]]],
                     output_mode: OutputMode = OutputMode.last_frame, softmax=False) -> Union[
    str, int, np.ndarray, List[Union[str, int, np.ndarray]]]:
    """Predicts results from "Deep learning-based automated TICI scoring: a timely proof-of-principle study" -
    experiment I & III, all networks are ensembled.
    Calls predict_series with proper initilization.
         args:
            config: Pathlike object pointing to a valid yaml to initialize model and dataloader or dictionary with
                    corresponding keyword arguments
            dcm_path: dicom paths to frontal and lateral view of an DSA-Series (order: [frontal, lateral]). Can be
                     a list of multiple series (list of lists).
            output_mode: valid values: 'last_frame' -> only the output of the last frame is returned
                                        'all_frames' -> the output of all frames will be returned
            softmax: True -> the softmax values are returned; False -> argmax and label_maping is applied to the
                     output
        return:
            score: tici-score predicted by the model either softmax or score based on mapping. If input is a list
                   of multiple series a list is returned"""
    return predict_series(config, dcm_path, output_mode, softmax, True)


'''This is just an example to demonstrate the workflow and give an overview over the used functions and output'''
if __name__ == "__main__":
    # dcm_path = ['/media/achilles/Storage/Xuchu_Liu/Workspace/DataSet/DiveMed/DB_images/uke_HH/20101025/4/uke_20101025_Ser4_AP_T0_M0_H3.dcm',
    #             '/media/achilles/Storage/Xuchu_Liu/Workspace/DataSet/DiveMed/DB_images/uke_HH/20101025/4/uke_20101025_Ser4_lat_T0_M0_H3.dcm']
    dcm_path = [['D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927141501723.1005.9.7334_0001_110000_16981757850000.dcm',
                'D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927141501723.1005.9.7334_0001_120000_16981757880001.dcm'],
    
                ['D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927141610851.1011.7.7724_0002_210000_16981758460067.dcm',
                'D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927141610851.1011.7.7724_0002_220000_16981758470068.dcm'],

                ['D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927141913504.1017.6.9962_0003_310000_1698175810005a.dcm',
                'D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927141913504.1017.6.9962_0003_320000_1698175812005b.dcm'],

                ['D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927142022419.1023.0.2145_0004_410000_16981758510074.dcm',
                'D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927142022419.1023.0.2145_0004_420000_16981758530075.dcm'],

                ['D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927142344551.1029.0.1961_0005_510000_169817586600a6.dcm',
                'D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927142344551.1029.0.1961_0005_520000_169817586700a7.dcm'],

                ['D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927144306038.1035.1.3942_0006_610000_16981758070052.dcm',
                'D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927144306038.1035.1.3942_0006_620000_16981758080053.dcm'],

                ['D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927144611541.1041.8.6391_0007_710000_169817586300a0.dcm',
                'D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927144611541.1041.8.6391_0007_720000_169817586400a1.dcm'],

                ['D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927144744302.1047.8.1153_0008_810000_1698175848006e.dcm',
                'D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927144744302.1047.8.1153_0008_820000_1698175849006f.dcm'],

                ['D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927152458791.1058.5.2718_0009_910000_1698175861009a.dcm',
                'D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927152458791.1058.5.2718_0009_920000_1698175861009b.dcm'],

                ['D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927154251706.1074.0.2912_0011_1110000_169817586900ac.dcm',
                'D:\\Xuchu_Liu\\Workspace\\DataSet\\DiveMed\\DB_images\\Rush\\1.2.392.200036.9116.3.1.12582476.20230927154251706.1074.0.2912_0011_1120000_169817587100ad.dcm']
    ]

    config = 'configuration.yml'
    print(predict_series(config, dcm_path))
    # for dsa in dcm_path:
    #     print(os.path.basename(dsa[0]))
    #     print(os.path.basename(dsa[1]))
    #     print(predict_series(config, dsa))
