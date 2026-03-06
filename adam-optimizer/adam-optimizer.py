import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)

    # Write code here
    m = m * beta1 + (1 - beta1) * grad
    v = v * beta2 + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    param = param - lr * (m_hat / (np.sqrt(v_hat) + eps))
    return (param, m, v)