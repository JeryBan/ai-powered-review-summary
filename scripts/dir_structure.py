'''
Creates the directories structure.
'''

from pathlib import Path

# data dir 
data_dir = Path('./data')
raw_dir = data_dir / 'raw'
processed_dir = data_dir / 'processed'
saved_models_dir = data_dir / 'saved_models'

# src dir
src_dir = Path('./src')
preprocessing_dir = src_dir / 'preprocessing'
models_dir = src_dir / 'models'
training_dir = src_dir / 'training'
evaluation_dir = src_dir / 'evaluation'
inference_dir = src_dir / 'inference'
utils_dir = src_dir / 'utils'

# create directories
data_dir.mkdir(exist_ok=True)
raw_dir.mkdir(exist_ok=True)
processed_dir.mkdir(exist_ok=True)
saved_models_dir.mkdir(exist_ok=True)

src_dir.mkdir(exist_ok=True)
preprocessing_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)
training_dir.mkdir(exist_ok=True)
evaluation_dir.mkdir(exist_ok=True)
inference_dir.mkdir(exist_ok=True)
utils_dir.mkdir(exist_ok=True)

# conf dir
conf_dir = Path('./config')
conf_dir.mkdir(exist_ok=True)

# scripts dir
scripts_dir = Path('./scripts')
scripts_dir.mkdir(exist_ok=True)

