import os

top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

## Directories
cache_dir = top_dir+"/cache"

## Local storage
USER = os.environ['USER']
vast_storage = f'/project01/ndcms/{USER}'

## Data Directories
data_dirs = {
    "MLNanoAODv9": "/project01/ndcms/atownse2/data/MLNanoAODv9",
    "skim_preselection": f"{top_dir}/data/skim_preselection",
    "skim_trigger_study": f"{top_dir}/data/skim_trigger_study",
}