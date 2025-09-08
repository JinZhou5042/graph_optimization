import os
import json

from typing import List, Union
from . import storage_config as stor
from . import dataset_info

import uproot
##

signal_processes = [
'BkkToGRadionToGGG',
'BkkToGRadionJetsToGGGJets'
]

class SignalPoint:

    def __init__(
        self,
        M_BKK: Union[int, float] = None,
        M_R: Union[int, float] = None,
        MOE: Union[int, float] = None,
        name: str = None,
        ):

        if name is not None:
            M_BKK, M_R = self.from_name(name)

        assert M_BKK is not None and (M_R is not None or MOE is not None), "Must specify M_BKK and either M_R or MOE"
        assert M_R is None or MOE is None, "Cannot specify both M_R and MOE"

        self.M_BKK = M_BKK

        if M_R is not None:
            self.M_R = M_R
            self.MOE = round(M_R/(M_BKK/2), 4)
        else:
            self.M_R = round((M_BKK/2)*MOE, 4)
            self.MOE = MOE

        # Remove decimal if integer
        if type(self.M_BKK) != int and self.M_BKK.is_integer():
            self.M_BKK = int(self.M_BKK)
        if type(self.M_R) != int and self.M_R.is_integer():
            self.M_R = int(self.M_R)

        self.Mass_Ratio = self.M_R/self.M_BKK


    def __eq__(self, other):
        return self.M_BKK == other.M_BKK and self.M_R == other.M_R

    def __hash__(self):
        return hash((self.M_BKK, self.M_R))

    def from_name(self, tag):
        import re

        M_BKK, M_R = None, None
        if 'M' not in tag or 'R0' not in tag:
            raise ValueError('Fragment does not contain mass point')
        else:
            M_BKK = float(
                re.search(r'M1-(\d+p\d+|\d+)', tag
                ).group(1).replace('p', '.'))

            M_R = float(
                re.search(r'R0-(\d+p\d+|\d+)', tag
                ).group(1).replace('p', '.'))
        
        return M_BKK, M_R

    @property
    def short_name(self):
        return f'{self.M_BKK}_{self.M_R}'
    
    @property
    def name(self):
        return f'M1-{self.M_BKK}_R0-{self.M_R}'.replace('.', 'p')

    # @property
    # def xs(self, signal_process):
    #     xs_pb = get_signal_xs(signal_process, self.M_BKK, self.M_R)['xs']
    #     xs_fb = xs_pb*1000
    #     return xs_fb
    
    # @property
    # def xs_error(self):
    #     xs_error_pb = get_signal_xs(self.M_BKK, self.M_R)['error']
    #     xs_error_fb = xs_error_pb*1000
    #     return xs_error_fb

gridpack_dir = '/cms/cephfs/data/store/user/atownse2/RSTriPhoton/gridpacks/'
gridpack_name = lambda M_BKK, M_R: f'BkkToGRadionToGGG_M1-{M_BKK}_R0-{M_R}_slc7_amd64_gcc10_CMSSW_12_4_8_tarball.tar.xz'

def get_signal_xs_pb(signal_process, M_BKK, M_R):

    signal_point = SignalPoint(M_BKK=M_BKK, M_R=M_R)
    _entry = f"{signal_process}_{signal_point.M_BKK}_{signal_point.M_R}"

    xs_dict = {}
    xs_file = stor.cache_dir+'/signal_xs_pb.json'
    if os.path.exists(xs_file):
        with open(xs_file, 'r') as f:
            xs_dict = json.load(f)
    
    if _entry in xs_dict:
        return xs_dict[_entry]
    
    gridpack = [f for f in os.listdir(gridpack_dir) if signal_point.name in f and signal_process in f][0]

    # Make workspace
    tmpdir = f'/tmp/{stor.USER}'
    if os.path.exists(tmpdir):
        os.system(f'rm -rf {tmpdir}')
    os.mkdir(tmpdir)
    os.chdir(tmpdir)

    # Unpack gridpack
    os.system(f'tar -xf {gridpack_dir+gridpack}')

    # Read cross section from .log file
    log_file = 'gridpack_generation.log'
    with open(log_file, 'r') as f:
        for line in f:
            if 'Cross-section :' in line:
                split_line = line.split()
                xs = float(split_line[2])
                error = float(split_line[4])
                if xs > 0:
                    xs_dict[_entry] = {'xs': xs, 'error': error}
    
    # Clean up
    os.system(f'rm -rf  {tmpdir}')

    # Save cross sections to yaml file
    with open(xs_file, 'w') as f:
        json.dump(xs_dict, f, indent=4)
    
    return xs_dict[_entry]