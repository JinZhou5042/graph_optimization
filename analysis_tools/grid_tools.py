# def query_xrootd(path, redirector=storage.redirector, top_dir=storage.top_dir):
#     cache_file = f'{top_dir}/cache/filelists/{path.replace("/","_")}_filelist.txt'
#     if not os.path.isfile(cache_file):
#         import XRootD.client
#         dirlist = []
#         fs = XRootD.client.FileSystem(redirector)
#         status, listing = fs.dirlist(path)
#         for f in listing:
#             dirlist.append(f.name)
#         with open(cache_file, 'w') as f:
#             f.write('\n'.join(dirlist))
    
#     with open(cache_file) as f:
#         dirlist = f.readlines()

#     return dirlist

# def query_das(query, outputfile):
#   print('Submitting query: ' + query)
#   os.system('dasgoclient --query "{}" >> {}'.format(query, outputfile))

# def get_local_filelist(glob_pattern):
#     '''Returns the filelist for the dataset'''
#     return [os.path.abspath(f) for f in glob.glob(glob_pattern+"*")]

# def get_das_filelist(data_location, redirector=storage.redirector, top_dir=storage.top_dir):
#     '''Returns the filelist for the dataset'''
#     filelist_dir = f'{top_dir}/cache/filelists'
#     if not os.path.isdir(filelist_dir):
#         os.makedirs(filelist_dir)

#     sample_name = data_location.replace('/', '_')[1:]
#     filelist_name = f'{filelist_dir}/{sample_name}_filelist.txt'
#     if not os.path.isfile(filelist_name):
#         query_das(f"file dataset={data_location}", filelist_name)
#     filelist = [redirector+f.replace('\n','') for f in open(filelist_name).readlines()]
#     return filelist

# ### For updating dataset config from DAS
# def sample_name_from_das(self, das_name): 
#     if self.isMC:
#         name = das_name.split('/')[1].split('_TuneCP5_')[0]
#         era_tag = r'RunIISummer20UL(\d{2})'
#     else:
#         name = das_name.split('/')[1]
#         era_tag = r'Run20(\d{2})'
    
#     match = re.search(era_tag, das_name)
#     if match is None:
#         raise ValueError(f'Could not find era in dataset name {das_name}')
                                                    
#     era = f"20{match.group(1)}"
#     if era == "2016":
#         if "APV" in das_name:
#             era += "APV"
#         elif "HIPM" in das_name:
#             era += "HIPM"

#     return f"{name}_{era}"