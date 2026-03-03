import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """    
    if len(seqs) == 0:
        return np.empty(0, 0)
        
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    res = np.full((len(seqs), max_len), pad_value)
    
    for i, seq in enumerate(seqs):
        copy_len = min(len(seq), max_len)
        res[i, :copy_len] = seq[:copy_len]
        
    return res