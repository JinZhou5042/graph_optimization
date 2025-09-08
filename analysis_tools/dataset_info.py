import os
import glob

from typing import Union, List

import json
import re

import analysis_tools.storage_config as storage
from analysis_tools import signal_info

cache_dir = f"{storage.cache_dir}/data"
if not os.path.isdir(cache_dir):
    os.makedirs(cache_dir)

all_data_formats = ['MiniAODv2', 'NanoAODv9']

years = ["2016preVFP", "2016postVFP", "2017", "2018"]

lumis_in_fb = {
    "2016preVFP" : 37.184259631, # fb^-1
    "2016postVFP" : 20.103495005,
    "2017" : 43.178270568,
    "2018": 62.448754676 
} # TODO check with someone that I used brilcalc correctly

dataset_name_include_all = {
    'data': ["DoubleEG", "EGamma"],
    'data_trigger_study': ["SingleElectron", "DoubleEG"],
    'signal': signal_info.signal_processes,
    'GJets': ["GJets_HT"],
}

def tag_dataset(dType, years, sample_name_or_filename):
    if dType not in dataset_name_include_all.keys():
        print(f"Warning: dType {dType} not recognized, tagging all datasets")
        return True
    
    pass_tag = any([name in sample_name_or_filename for name in dataset_name_include_all[dType]])
    pass_year = any([year in sample_name_or_filename for year in years])

    return pass_tag and pass_year

def get_years_from_era(era):
    if "," in era: return era.split(',')
    elif era in years: return [era]
    elif era == "Run2" : return years
    else: raise ValueError(f'era {era} not recognized')

def get_n_events_in_root_file(file_name, tree_name='Events'):
    import uproot
    f = uproot.open(file_name)
    return f[tree_name].numentries

class Dataset:
    """Class for handling dataset information and access"""
    file_extension = '.root'

    def __init__(
            self,
            dType: str,
            sample_name: str,
            data_format: str,
            storage_base: str = None,
            update_dataset_info: bool = False,
            **kwargs
            ):
        """
        Initialize a dataset object

        Args:
            dType (str): Data type of the Dataset (e.g. data, GJets, signal, etc.)
            sample_name (str): Name of the sample (e.g. GJets_HT-40To100_2018)
            data_format (str): Format of the Dataset (e.g. MiniAODv2, NanoAODv9, etc.)
            storage_base (str): Base storage directory for the dataset (if stored locally)
        """

        self.dType = dType
        self.isMC = dType != 'data'

        self.sample_name = sample_name
        self.data_format = data_format

        if dType == 'signal':
            self.signal_point = signal_info.SignalPoint(name=sample_name)
            self.M_BKK = self.signal_point.M_BKK
            self.M_R = self.signal_point.M_R
            self.signal_process = self.sample_name.split('_')[0]
        
        self.kwargs = kwargs

    @property
    def name(self):
        return f'{self.sample_name}_{self.data_format}'

    @property
    def n_events(self):

        if hasattr(self, 'events'):
            print(f"Warning: If you have already applied a selection, this will not represent the total events in the file")
            return len(self.events)

        cache_file = f'{cache_dir}/{self.dType}.json'
        if "reset_cache" in self.kwargs:
            pass
        else:
            if os.path.exists(cache_file):
                return json.load(open(cache_file))[self.name]
        
        # Count Events 
        n_events = 0
        for f in self.files:
            if not 'root' in f:
                raise NotImplementedError(f"File format {f} not supported")
            n_events += get_n_events_in_root_file(f)
        
        # Update cache
        old_cache = json.load(open(cache_file)) if os.path.exists(cache_file) else {}
        old_cache[self.sample_name] = n_events
        with open(cache_file, 'w') as f:
            json.dump(old_cache, f, indent=4)
            
        return n_events

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif hasattr(self, 'get'):
            return self.get(key)
        raise KeyError(f'Key {key} not found in {self.__class__}')

    @property
    def year(self):
        for y in years:
            if y in self.sample_name:
                return y
        print(f"Warning: Year not found in dataset tag {self.sample_name}, defaulting to 2018")
        return "2018"
        # raise ValueError(f'Year not found in dataset tag {self.sample_name}')

    @property
    def era(self):
        return self.year

    @property
    def files(self):
        if hasattr(self, '_files'): return self._files

        file_dir = storage.data_dirs[self.data_format]
        file_tag = f"{self.sample_name}_{self.data_format}"

        files = []
        for f in os.listdir(file_dir):
            if file_tag in f:
                files.append(f"{file_dir}/{f}")

        self._files = files
        return self._files

class Datasets:
    """Class for intuitively and flexibly acessing multiple datasets"""

    dataset_class = Dataset

    def __init__(
            self,
            dType: str,
            era: str,
            data_format: str,
            subset: Union[str, List[str], List[Dataset]] = None,
            **kwargs
            ):
        """
        Initialize a Datasets object

        Args:
            dTypes (str or ): Data types of the Datasets (e.g. data, GJets, signal, etc.)
            era (str): Era of the Datasets (e.g. 2018, Run2, etc.)
            data_format (str): Format of the Datasets (e.g. MiniAODv2, NanoAODv9, etc.)
            subset (str, list): Subset of datasets to include can be a list of strings
                or a list of Dataset objects. String specification can be a comma separated,
                and should follow the pattern "dType/sample_name" (e.g. "GJets_HT-40To100_2018")
            storage_base (str): Base storage directory for the datasets (if stored locally)
        """

        self.dType = dType
        self.era = era
        self.years = get_years_from_era(era)

        self.data_format = data_format

        if subset is not None and not isinstance(subset, list):
            subset = [subset]

        self.subset = subset

        self.kwargs = kwargs

        self.set_up_datasets()

    @property
    def name(self):
        if self.dType == 'signal':
            if len(self.signal_points) == 1:
                return f"{'-'.join(self.signal_processes)}_{self.signal_point.name}_{self.era}_{self.data_format}"
        return f"{self.dType}_{self.era}_{self.data_format}"

    @property
    def signal_points(self):
        assert self.dType == 'signal', "Signal points only available for signal datasets"
        return set([d.signal_point for d in self.datasets.values() if hasattr(d, 'signal_point')])

    @property
    def signal_point(self):
        assert self.dType == 'signal', "Signal point only available for signal datasets"
        signal_points = self.signal_points
        if len(signal_points) == 1:
            return signal_points.pop()
        elif len(signal_points) > 1:
            raise ValueError(f"Multiple signal points found in Datasets: {signal_points}")

    @property
    def signal_processes(self):
        assert self.dType == 'signal', "Signal processes only available for signal datasets"
        return set([d.signal_process for d in self.datasets.values() if hasattr(d, 'signal_process')])

    @property
    def signal_process(self):
        assert self.dType == 'signal', "Signal process only available for signal datasets"
        signal_processes = set([d.signal_process for d in self.datasets.values() if hasattr(d, 'signal_process')])
        if len(signal_processes) == 1:
            return signal_processes.pop()
        elif len(signal_processes) > 1:
            # raise ValueError(f"Multiple signal processes found in Datasets: {signal_processes}")
            # print(f"Warning: Multiple signal processes found in Datasets: {signal_processes}")
            # print(f"Returning first tag")
            return signal_processes.pop()

    @property
    def files(self):
        return [f for d in self.datasets.values() for f in d.files]

    def __iter__(self):
        return iter(self.datasets.values())

    def __len__(self):
        return len(self.datasets.keys())

    def get(self, key):
        raise NotImplementedError(f'Getting {key} from {self.dataset_class} not implemented') 

    def copy(self, **kwargs):
        return type(self)(
            self.dType, self.era, self.data_format,
            **self.kwargs,
            **kwargs
        )


    def __getitem__(self, key):
        """Return a subset of the datasets or other objects through self.get()"""

        if not isinstance(key, list):
            key = [key]

        # print(f"Getting {key} from {self.__class__}")
        # for d in self.datasets.keys():
        #     print(d)
        #     print(key[0] in d)
        #     print()

        subset = []
        for k in key:
            if isinstance(k, self.dataset_class):
                subset.append(k)
            elif isinstance(k, int):
                subset.append(self.datasets.values()[k])
            elif isinstance(k, slice):
                subset.extend(list(self.datasets.values())[k])
            elif isinstance(k, str):
                subset += [d for name, d in self.datasets.items() if k in name or k == name]
            elif isinstance(k, signal_info.SignalPoint):
                subset += [d for d in self.datasets.values() if d.signal_point == k]
            else:
                raise ValueError(f'Class Datasets does not support key type {type(k)}')
        
        if len(subset) == 0:
            # return self.get(key)
            raise KeyError(f'Key {key} not found in {self.__class__}')
        elif len(subset) == 1:
            return subset[0]
        else:
            # return self.copy(subset=subset)
            copy =  self.copy()
            copy.datasets = {d.name: d for d in subset}
            return copy
        

    def set_up_datasets(self):

        self.datasets = {}
        for f in os.listdir(storage.data_dirs[self.data_format]):
            if tag_dataset(self.dType, self.years, f):
                if self.dType == 'signal': #TODO there has to be a more intuitive way to specify this
                    n_underscores_to_keep = 4
                elif self.dType == 'data':
                    n_underscores_to_keep = 3
                elif self.dType == 'GJets':
                    n_underscores_to_keep = 3

                sample_name = "_".join(f.split('_')[:n_underscores_to_keep])

                _dataset = self.dataset_class(self.dType, sample_name, self.data_format, **self.kwargs)
                self.datasets[_dataset.name] = _dataset

        if self.subset is not None:
            self.datasets = {d.name: d for d in self[self.subset]}


#### End Data Tools

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='Initialize the dataset config file')
    parser.add_argument('--update', '-u', action='store_true', help='Update the datasets from DAS')
    parser.add_argument('--dType', '-d', type=str, default=None, help='Data type of the Dataset (e.g. data, GJets, QCD, etc.)')
    parser.add_argument('--format', '-f', type=str, default=None, help='Format of the Dataset (e.g. MiniAODv2, NanoAODv9, etc.)')

    args = parser.parse_args()
    
    if args.update:
        if args.dType is None or args.format is None:
            raise ValueError('dType and format must be specified when updating from DAS')
        # datasets = Datasets(args.dType, update=True).datasets
        raise NotImplementedError('Updating from DAS not implemented')
