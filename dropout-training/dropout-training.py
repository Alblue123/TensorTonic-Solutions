import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code he
    x = np.array(x)
    rand = rng.random((x.shape))
    threshold = 1 - p
    masks = np.where(rand < threshold, 1/threshold, 0)
    res = x * masks
    return (res, masks)