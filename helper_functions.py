import random
import numpy as np
import torch

def set_seed(seed: int = 666):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_metrics(metrics_dict):
    cleaned = {}

    for key, values in metrics_dict.items():
        cleaned_values = []
        for v in values:
            if isinstance(v, torch.Tensor):
                cleaned_values.append(v.item())
            else:
                cleaned_values.append(float(v))
        cleaned[key] = cleaned_values

    return cleaned

