import sys
import os
import warnings
import math
import time

import dask
import hashlib
import json

from ndcctools.taskvine.dask_executor import  daskvine_merge

import numpy as np
import awkward as ak
import dask_awkward as dak

top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(top_dir)

from analysis_tools import dataset_info as di
from analysis_tools import storage_config as storage
from analysis_tools.storage_config import cache_dir
from analysis_tools import signal_info as si

from analysis import selections as sel
from analysis import variables as var
from analysis import calculations as calc
from dask.optimization import cull
from dask.optimization import inline_functions

def load_skims(
    dType,
    selection_tag='preselection',
    era='Run2',
    **kwargs,
    ):
    
    skims = SkimmedDatasets(dType, era, f"skim_{selection_tag}", **kwargs)
    return skims

def get_N_events(dataset):
    dType = dataset.dType

    if dType == 'data':
        raise ValueError("Number of events for data not needed")
    
    n_events_dict = {}

    cache_file = f"{cache_dir}/{dType}_n_events.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            n_events_dict = json.load(f)
        
        if dataset.name in n_events_dict:
            return n_events_dict[dataset.name]

    import uproot

    print(f"Getting number of events for {dataset.name}")
    n_events = 0
    for file in dataset.files:
        with uproot.open(file) as f:
            n_events += f['Events'].num_entries

    n_events_dict[dataset.name] = n_events
    
    with open(cache_file, 'w') as f:
        json.dump(n_events_dict, f, indent=4)

    return n_events_dict[dataset.name]

class SkimmedDataset(di.Dataset):
    file_extension = '.parquet'
    # scaled = False

    @property
    def file(self):
        return self.files[0]

    @property
    def events(self):
        if hasattr(self, '_events'):
            return self._events
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from analysis_tools.coffea_utils import MLNanoAODSchema

        if len(self.files) == 0:
            raise ValueError("No files found for dataset")
        elif len(self.files) > 1:
            raise ValueError("More than one file found for dataset")

        with np.errstate(divide='ignore', invalid='ignore'):
            events = ak.from_parquet(
                self.files[0],
                behavior = MLNanoAODSchema.behavior(),
                )

            for field in events.fields:
                events[field] = ak.with_name(events[field], field)

            # Calculate quantities and objects
            events['Candidates'] = calc.candidates(events)
            events = events[ak.num(events.Candidates) > 0]

            events['Candidate'] = calc.candidate(events.Candidates)
            events['weight'] = ak.ones_like(events.run, dtype=np.float32)


            # Make selections if necessary
            if 'apply_selection' in self.kwargs:
                events = sel.selections[self.kwargs['apply_selection']](events)

        self._events = events
        if self.isMC:
            self.scale_mc()

        return self._events

    def cut(self, cut):
        var_name, op, value = cut.split()

        if op == ">":
            mask = self.get(var_name) > float(value)
        elif op == "<":
            mask = self.get(var_name) < float(value)
        elif op == ">=":
            mask = self.get(var_name) >= float(value)
        elif op == "<=":
            mask = self.get(var_name) <= float(value)

        self._events = self._events[mask]
        # Recalculate quantities and objects
        self._events['Candidates'] = calc.candidates(self._events)
        self._events = self._events[ak.num(self._events.Candidates) > 0]
        self._events['Candidate'] = calc.candidate(self._events.Candidates)
    
    def apply_selection(self, selection):
        return sel.selections[selection](self._events)

    def get(self, var_name):
        # if var_name == 'weight' and not self.scaled:
        #     raise ValueError("Must scale MC first")
        return var.get(self.events, var_name)

    def get_n_nanoaod_events(self):
        n_total = get_N_events(di.Dataset(self.dType, self.sample_name, "MLNanoAODv9") )
        return n_total

    def scale_mc(self, xs_factor=1.0):
        if not self.isMC:
            raise ValueError("Cannot scale data")
        # if self.scaled:
        #     raise ValueError("Already scaled")

        if '_events' not in self.__dict__:
            self.events

        if self.dType == 'signal':
            # xs_pb = self.signal_point.xs(self.signal_process)
            xs_pb = si.get_signal_xs_pb(self.signal_process, self.signal_point.M_BKK, self.signal_point.M_R)['xs']
        elif self.dType == 'GJets':
            xs_file = f"{cache_dir}/GJets_xs_pb.json"
            with open(xs_file, 'r') as f:
                xs_dict = json.load(f)
            xs_pb = xs_dict[self.name.replace(f"_{self.data_format}", '')]

        xs_fb = xs_pb * 1000 * xs_factor

        n_total = self.get_n_nanoaod_events()

        lumiscale = di.lumis_in_fb[self.year]*xs_fb/n_total

        self._events['weight'] = lumiscale * self._events['weight']
        # self.scaled = True
    
    def write(self, outpath):
        ak.to_parquet(self._events, outpath)

class SkimmedDatasets(di.Datasets):
    dataset_class = SkimmedDataset

    def get(self, var_name):
        return var.get(self.events, var_name)

    def cut(self, cuts):
        if isinstance(cuts, str):
            cuts = [cuts]
        
        mask = None
        for cut in cuts:
            var_name, op, value = cut.split()

            if op == ">":
                _mask = self.get(var_name) > float(value)
            elif op == "<":
                _mask = self.get(var_name) < float(value)
            elif op == ">=":
                _mask = self.get(var_name) >= float(value)
            elif op == "<=":
                _mask = self.get(var_name) <= float(value)
            
            if mask is None:
                mask = _mask
            else:
                mask = mask & _mask

        self._events = self._events[mask]
        # Recalculate quantities and objects
        self._events['Candidates'] = calc.candidates(self._events)
        self._events = self._events[ak.num(self._events.Candidates) > 0]
        self._events['Candidate'] = calc.candidate(self._events.Candidates)
    
    def scale_mc(self, xs_factor=1.0):
        for d in self:
            d.scale_mc(xs_factor)
    
    def get_n_nanoaod_events(self):
        n_total = 0
        for d in self:
            n_total += d.get_n_nanoaod_events()
        return n_total

    def apply_selection(self, selection):
        return sel.selections[selection](self._events)
        
    @property
    def events(self):
        if any([d.dType == 'signal' for d in self]):
            signal_points = [d.signal_point for d in self]
            if any([sp != signal_points[0] for sp in signal_points]):
                raise ValueError("Cannot concatenate MC datasets with different signal points")
        
    
        if hasattr(self, '_events'):
            return self._events

        # Concatenate events from all datasets
        all_events = []
        for d in self:
            all_events.append(d.events)
        self._events = ak.concatenate(all_events, axis=0)

        return self._events

@daskvine_merge
def merge(*events):
    import awkward as ak
    events = ak.concatenate(events)
    return events

def ak_write(events, path):
    events
    


def write_ak(events, path):
    import awkward as ak
    ak.to_parquet(events, path)
    return path

def read_and_write(dir_path, *paths, filename=None):
    import awkward as ak
    import hashlib

    if filename:
        path = f'{dir_path}/{filename}'
    else:
        name = '-'.join(paths)
        filename = hashlib.md5(name.encode()).hexdigest()
        path = f'{dir_path}/{filename}.parquet'

    events = ak.from_parquet(paths)
    ak.to_parquet(events, path)
    return path

 
class Skimmer:

    def __init__(
        self,
        dType, analysis_region, era,
        **test_args,
        ):
        from analysis_tools.coffea_utils import CoffeaDatasets

        self.dType = dType
        self.analysis_region = analysis_region
        self.era = era

        self.input_datasets = CoffeaDatasets(dType, era, 'MLNanoAODv9')

        self.output_format = 'skim_'+self.analysis_region
        self.output_dir = storage.data_dirs[self.output_format]
        
        self.test_args = test_args
        if test_args:
            self.output_dir += '_test'

        self.tmp_dir = f"{self.output_dir}/tmp"
        if os.path.exists(self.tmp_dir):
            os.system(f"rm -rf {self.tmp_dir}")
        os.makedirs(self.tmp_dir)

        if 'n_test_datasets' in test_args:
            self.input_datasets = self.input_datasets[:test_args['n_test_datasets']]

    def write_partitions_auto(self, fileset, optimize_graph=False):
        from coffea.nanoevents import NanoEventsFactory
        from coffea.util import decompress_form
        from analysis_tools.coffea_utils import MLNanoAODSchema

        graph = {}
        self.npartitions = {}
        dsets = 0
        for dataset_name, info in fileset.items():
            dsets += 1
            events = NanoEventsFactory.from_root(
                info['files'],
                schemaclass=MLNanoAODSchema,
                known_base_form=ak.forms.from_json(decompress_form(info['form']))
            ).events()

            if not os.path.exists(f"{self.tmp_dir}/{dataset_name}/"):
                os.mkdir(f"{self.tmp_dir}/{dataset_name}/")

            if self.analysis_region == 'preselection':
                events = sel.preselection(events, info['year'])
            elif self.analysis_region == 'trigger_study':
                events = sel.trigger_study(events, info['year'])
            else:
                raise ValueError(f"Analysis region {self.analysis_region} not recognized.")
            
            self.npartitions[dataset_name] = events.npartitions
            print(events.npartitions)
            partitions = []
            merge_size = events.nnpartitions
            jump_size = math.ceil(events.npartitions/merge_size)
            for i in range(jump_size):
                p = events.partitions[i*merge_size:i*merge_size+merge_size]
                partitions.append(p)

            x = dask.delayed(merge)(*partitions)
            x = dask.delayed(ak_write)(x, f"{self.tmp_dir}/{dataset_name}/final.parquet")

            graph[f"{dataset_name}_write"] = x
            if dsets >= 10:
                break
        return graph

    def write_partitions(self, fileset, optimize_graph=False):
        from coffea.nanoevents import NanoEventsFactory
        from coffea.util import decompress_form
        from analysis_tools.coffea_utils import MLNanoAODSchema

        graph = {}
        self.npartitions = {}
        dsets = 0
        for dataset_name, info in fileset.items():
            dsets += 1
            events = NanoEventsFactory.from_root(
                info['files'],
                schemaclass=MLNanoAODSchema,
                known_base_form=ak.forms.from_json(decompress_form(info['form']))
            ).events()

            if self.analysis_region == 'preselection':
                events = sel.preselection(events, info['year'])
            elif self.analysis_region == 'trigger_study':
                events = sel.trigger_study(events, info['year'])
            else:
                raise ValueError(f"Analysis region {self.analysis_region} not recognized.")
            
            self.npartitions[dataset_name] = events.npartitions
            print(events.npartitions)
            delayed  = []
            merge_size = 2
            jump_size = math.ceil(events.npartitions/merge_size)
            for i in range(jump_size):
                path = f"{self.tmp_dir}/{dataset_name}/part{i}.parquet"
                if not os.path.exists(f"{self.tmp_dir}/{dataset_name}/"):
                    os.mkdir(f"{self.tmp_dir}/{dataset_name}/")
                d = events.partitions[i*merge_size:i*merge_size+merge_size]
                delayed.append(dask.delayed(write_ak)(d, path))
            while(len(delayed) > 1):
                to_merge = [delayed.pop(0) for i in range(merge_size) if delayed]
                if delayed:
                    d = dask.delayed(read_and_write)(f'{self.tmp_dir}/{dataset_name}/', *to_merge)
                else:
                    d = dask.delayed(read_and_write)(f'{self.tmp_dir}/{dataset_name}/', *to_merge, filename='final.parquet')
                delayed.append(d)
            graph[f"{dataset_name}_write"] = delayed[0]
            if dsets >= 10:
                break
        return graph

    def merge_all_partitions(self, fileset, partition_merge_size=4,):
        import hashlib

        graph = {}
        for dataset_name, info in fileset.items():
            final_tag = dataset_name.replace('MLNanoAODv9', self.output_format)

            # Then merge partitions
            file_path = lambda filename: f"{self.tmp_dir}/{dataset_name}/{filename}"
            batch_num = lambda filename: filename.split('.')[0].replace("part","")

            written_partitions = [[file_path(f), batch_num(f)] for f in os.listdir(f"{self.tmp_dir}/{dataset_name}") if f.endswith('.parquet')]

            if len(written_partitions) == 1:
                old_filename = written_partitions[0][0]
                new_filename = f"{self.output_dir}/{final_tag}.parquet"
                os.rename(old_filename, new_filename)
                continue
            
            while len(written_partitions) > 1:
                n_partitions = min(partition_merge_size, len(written_partitions))
                to_merge = [written_partitions.pop(0) for _ in range(n_partitions) if len(written_partitions)>0]
                futures = [x[0] for x in to_merge]
                tags = [x[1] for x in to_merge]

                if len(written_partitions) > 0:
                    name = '-'.join(tags)
                    hashstring = hashlib.md5(name.encode())
                    tag = hashstring.hexdigest()
                    final = False
                    _dir = f"{self.tmp_dir}/{dataset_name}"
                else:
                    tag = final_tag
                    final = True
                    _dir = self.output_dir
                
                future_file_path = self.merge_partitions(
                    futures,
                    f"{_dir}/{tag}.parquet",
                    final=final)

                written_partitions.append([future_file_path, tag])
                graph[tag] = future_file_path

        return graph

    def skim(
        self,
        step_size = 250_000,
        use_dask_vine=False,
        use_local_cluster=False,
        manager_port=9135,
        optimize_graph=False,
        partition_merge_size=2,
        print_time=False,
        no_skim=False,
        **test_args
        ):

        import time
        import os 

        scheduler = None
        if use_dask_vine:
            from ndcctools.taskvine import DaskVine
            run_info_path = f"{cache_dir}/vine-run-info"
            if not os.path.exists(run_info_path):
                os.makedirs(run_info_path)

            m = DaskVine(
                manager_port,
                name="rstriphoton",
                run_info_path=run_info_path,
                )
            m.tune('wait-for-workers', 1)

            _di = m.declare_file(f"{top_dir}/analysis_tools/dataset_info.py", cache=True)
            _si = m.declare_file(f"{top_dir}/analysis_tools/signal_info.py", cache=True)
            _sc = m.declare_file(f"{top_dir}/analysis_tools/storage_config.py", cache=True)
            _coffea = m.declare_file(f"{top_dir}/analysis_tools/coffea_utils.py",cache=True)
            _vars = m.declare_file(f"{top_dir}/analysis/variables.py", cache=True)
            _sel = m.declare_file(f"{top_dir}/analysis/selections.py", cache=True)
            _calc = m.declare_file(f"{top_dir}/analysis/calculations.py", cache=True)
            _pl = m.declare_file(f"{top_dir}/analysis/plotting.py", cache=True)

            extra_files = {
                _di : "analysis_tools/dataset_info.py",
                _si : "analysis_tools/signal_info.py",
                _sc : "analysis_tools/storage_config.py",
                _coffea : "analysis_tools/coffea_utils.py",
                _vars : "analysis/variables.py",
                _sel : "analysis/selections.py",
                _calc : "analysis/calculations.py",
                _pl : "analysis/plotting.py",
            }

            scheduler = m.get
    
        # Preprocess
        t1 = time.time()
        filesets = self.input_datasets.filesets(
            step_size,
            scheduler=scheduler,
            **self.test_args)
        if print_time: print(f"Preprocess: {time.time()-t1:.2f}")
        # Write partitions
        t2 = time.time()
        #self.write = self.write_partitions_auto(filesets)
        self.write = self.write_partitions(filesets)
        print(f"type = {type(self.write)}")
        import cloudpickle
        with open("rstriphoton.pkl", "wb") as f:
            cloudpickle.dump(self.write, f)
        if print_time: print(f"Create write graph: {time.time()-t2:.2f}")

        if no_skim: return
        t3 = time.time()

        if use_dask_vine:
            x = dask.compute(
                self.write,
                scheduler=scheduler,
                prune_files=False,
                prune_depth=2,
                reconstruct=True,
                merge_size=3,
                resources={"cores": 1},
                resources_mode=None,
                lazy_transfers=True,
                extra_files=extra_files,
            )
            print(x)
        elif use_local_cluster:
            from dask.distributed import LocalCluster, Client
            cluster = LocalCluster(n_workers=8)
            with Client(cluster) as client:
                dask.compute(self.write, scheduler=client)
        else:
            dask.compute(self.write)
        if print_time: print(f"Write Partitions: {time.time()-t3:.2f}")
        if print_time: print(f"Total: {time.time()-t1:.2f}")
        exit(1)

        # Remove temporary directory
        os.system(f"rm -rf {self.tmp_dir}")

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dType", help="Data type to process: data or signal")
    parser.add_argument("--analysis_region", default='preselection', help="Analysis region to process: None or preselection")
    parser.add_argument("--era", type=str, default="Run2", help="Year(s) of data to process")
    parser.add_argument("--step_size", type=int, default=250_000, help="Chunk size for processing")
    parser.add_argument("--use_dask_vine", "-dv", action="store_true", help="Use dask vine for processing")
    parser.add_argument("--use_local_cluster", "-lc", action="store_true", help="Use local cluster for processing")
    parser.add_argument("--optimize_graph", action="store_true", help="Optimize the dask graph")
    parser.add_argument("--print_time", type=bool, default=True, help="Print timing information")
    parser.add_argument("--test", "-t", action="store_true", help="Run in test mode")
    parser.add_argument("--test_step_size", type=int, default=50, help="Step size for test mode")
    parser.add_argument("--n_test_files", type=int, default=2, help="Number of files to process in test mode, None for all files")
    parser.add_argument("--n_test_steps", type=int, default=2, help="Number of steps to process in test mode, None for all steps")
    parser.add_argument("--n_test_datasets", type=int, default=2, help="Number of datasets to process in test mode, None for all datasets")
    parser.add_argument("--no_skim", action="store_true", help="Do not run the skim, only preprocess and create graph")
    args = parser.parse_args()

    dType = args.dType
    analysis_region = args.analysis_region
    era = args.era
    step_size = args.step_size
    optimize_graph = args.optimize_graph
    use_dask_vine = args.use_dask_vine

    no_skim = args.no_skim
    print_time = args.print_time

    test = args.test
    test_args = {}

    if test:
        print_time = True
        step_size = args.test_step_size
        if args.n_test_files is not None:
            test_args['n_test_files'] = args.n_test_files
        if args.n_test_steps is not None:
            test_args['n_test_steps'] = args.n_test_steps
        if args.n_test_datasets is not None:
            test_args['n_test_datasets'] = args.n_test_datasets

    skimmer = Skimmer(dType, analysis_region, era, **test_args)
    skimmer.skim(
        step_size,
        use_dask_vine=use_dask_vine,
        use_local_cluster=args.use_local_cluster,
        optimize_graph=optimize_graph,
        print_time=print_time,
        no_skim=no_skim,
    )
