import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    # sin(pos/base^2i/dmodel)
    pe = np.zeros((seq_len, d_model), dtype=float)
    pos = np.arange(seq_len).reshape(seq_len, 1)
    if d_model % 2 == 0: 
        freq = d_model // 2 
    else: freq = d_model // 2 + 1
        
    i = np.arange(freq).reshape(1, freq)
    freqs =  1 / base**(2*i/d_model)
    angles = pos * freqs
    pe[:,0::2] = np.sin(angles)
    pe[:,1::2] = np.cos(angles[:,:d_model//2])
    return pe
    
    