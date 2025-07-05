import random
import numpy as np
import torch

def set_seed(seed: int = 666):

    """
    Sets the random seed across Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int, optional): The seed value to use. Defaults to 666.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_metrics(metrics_dict):

    """
    Returns new dictionary with the same keys and Tensor metric values converted to Python floats.

    Args:
        metrics_dict (dict): Dictionary where keys are metric names and values are lists 
            containing numbers or PyTorch tensors.

    Example:
        Input: {'loss': [tensor(0.5), tensor(0.3)], 'acc': [0.8, tensor(0.9)]}
        Output: {'loss': [0.5, 0.3], 'acc': [0.8, 0.9]}
    """

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


def reorder_and_merge_classes(class_map, y_true, y_pred, y_prob, to_merge):

    """
    Reorders class indices and merges specified classes into a single 'Other' class.

    Args:
        class_map (dict): Mapping from class names to original indices.
        y_true (np.ndarray): Array of true labels with original indices.
        y_pred (np.ndarray): Array of predicted labels with original indices.
        y_prob (np.ndarray): Array of predicted probabilities of shape (n_samples, n_original_classes).
        to_merge (list): List of class names to merge into 'Other'.

    Returns:
        tuple:
            - new_class_map (dict): Mapping from class names (with 'Other') to new indices.
            - new_y_true (np.ndarray): True labels remapped to new indices.
            - new_y_pred (np.ndarray): Predicted labels remapped to new indices.
            - new_y_prob (np.ndarray): Probability array reshaped to (n_samples, n_new_classes),
              with probabilities of merged classes combined and normalized.

    Notes:
        - Probability vectors are summed across merged classes and renormalized to sum to 1, to avoid
          floating point error downstream.
    """

    class_map_rev = {v: k for k, v in class_map.items()} # Inverse class map

    # Create new class map with main classes + other
    to_keep = sorted([cls for cls in class_map if cls not in to_merge])
    new_class_map = {cls: idx for idx, cls in enumerate(to_keep)}

    other_index = len(to_keep)
    new_class_map['Other'] = other_index

    # Map old indices to new indices
    orig_max_index = max(class_map.values())
    orig_to_new_map = np.zeros(orig_max_index + 1, dtype = int)

    for orig_idx, cls in class_map_rev.items():
        if cls in to_merge:
            orig_to_new_map[orig_idx] = other_index
        else:
            orig_to_new_map[orig_idx] = new_class_map[cls]

    # Map labels
    new_y_true = orig_to_new_map[y_true]
    new_y_pred = orig_to_new_map[y_pred]

    # Map prediction probabilities
    n_samples = y_prob.shape[0]
    new_n_classes = len(new_class_map)
    new_y_prob = np.zeros((n_samples, new_n_classes))
    
    for orig_idx, cls in class_map_rev.items():
        new_idx = new_class_map['Other'] if cls in to_merge else new_class_map[cls]
        new_y_prob[:, new_idx] += y_prob[:, orig_idx]

    # Need to normalize due to floating-point precision
    y_prob_sums = new_y_prob.sum(axis = 1, keepdims = True)
    new_y_prob = new_y_prob / y_prob_sums

    return new_class_map, new_y_true, new_y_pred, new_y_prob

