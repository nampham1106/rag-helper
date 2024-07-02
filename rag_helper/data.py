from datasets import load_dataset, Dataset
from tqdm import tqdm
import json

def get_labels(file_path):
    with open(file_path, 'r') as f:
        labels = []
        for line in f:
            labels.append(json.loads(line))
    return labels