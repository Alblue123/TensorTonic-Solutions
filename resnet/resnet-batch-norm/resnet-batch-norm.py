import numpy as np

class BatchNorm:
    """Batch Normalization layer."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply batch normalization.
        """
        # YOUR CODE HERE
        if training:
           mean_b = np.mean(x, axis=0)
           var_b = np.var(x, axis=0)
           self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_b
           self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_b
           x_hat = (x - mean_b) / (np.sqrt(var_b + self.eps))
           return self.gamma * x_hat + self.beta
            
        x_hat = (x - self.running_mean) / (np.sqrt(self.running_var + self.eps))
        return self.gamma * x_hat + self.beta

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)

def post_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Post-activation ResNet block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Uses x @ W for "convolution" (simplified as linear transform).
    """
    # YOUR CODE HERE
    return relu(bn2.forward(relu(bn1.forward(x @ W1)) @ W2) + x)

def pre_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Pre-activation ResNet block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    This ordering often works better for very deep networks.
    """
    # YOUR CODE HERE
    return relu(bn2.forward(relu(bn1.forward(x)) @ W1)) @ W2 + x
