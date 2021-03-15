import numpy as np

# 1D utilities.


def convert_int_rep_to_onehot(state, vector_size):
    s = np.zeros(vector_size)
    s[state] = 1
    return s


def convert_onehot_to_int(state):
    if type(state) is not np.ndarray:
        state = np.array(state)
    return state.argmax().item()

