import torch
from settings import PRECOMPUTED_PATH

def load_precomputed(path=PRECOMPUTED_PATH):
    data = torch.load(path)
    return data["link"], data["n"], data["features"]
