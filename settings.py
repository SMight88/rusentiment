import os
from pathlib import Path

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print('ROOT_DIR =', ROOT_DIR)

DATA_DIR = Path(ROOT_DIR  + '/data/')
EMBEDDINGS_DIR = DATA_DIR.joinpath('embeddings/')
PATH_TO_EMBEDDINGS = EMBEDDINGS_DIR.joinpath('fasttext.min_count_100.vk_posts_all_443550246.300d.vec')
PATH_TO_SCALER = DATA_DIR.joinpath('scaler.pkl')
PATH_TO_MODEL = DATA_DIR.joinpath('mlp.h5')
PATH_TO_MLB = DATA_DIR.joinpath('mlb.pkl')