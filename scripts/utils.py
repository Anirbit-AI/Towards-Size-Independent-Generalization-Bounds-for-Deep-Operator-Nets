import numpy as np


def save_checkpoint(model_params, filename):
    np.savez(filename, **model_params)

def load_checkpoint(filename):
    data = np.load(filename)
    model_params = {key: data[key] for key in data}
    return model_params
