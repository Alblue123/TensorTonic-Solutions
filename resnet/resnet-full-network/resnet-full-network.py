import numpy as np

def relu(x):
    return np.maximum(0, x)

class BasicBlock:
    def __init__(self, in_ch, out_ch, downsample=False):
        self.W1 = np.random.randn(in_ch, out_ch) * 0.01
        self.W2 = np.random.randn(out_ch, out_ch) * 0.01
        self.W_proj = np.random.randn(in_ch, out_ch) * 0.01 if in_ch != out_ch or downsample else None

    def forward(self, x):
        out = x @ self.W1
        out = relu(out)
        out = out @ self.W2
        if self.W_proj is not None:
            out = out + x @ self.W_proj
        else:
            out = out + x
        return relu(out)

class ResNet18:
    def __init__(self, num_classes=10):
        self.conv1 = np.random.randn(3, 64) * 0.01

        self.layer1 = [BasicBlock(64, 64), BasicBlock(64, 64)]
        self.layer2 = [BasicBlock(64, 128, True), BasicBlock(128, 128)]
        self.layer3 = [BasicBlock(128, 256, True), BasicBlock(256, 256)]
        self.layer4 = [BasicBlock(256, 512, True), BasicBlock(512, 512)]

        self.fc = np.random.randn(512, num_classes) * 0.01

    def forward(self, x):
        out = relu(x @ self.conv1)

        for block in self.layer1:
            out = block.forward(out)
        for block in self.layer2:
            out = block.forward(out)
        for block in self.layer3:
            out = block.forward(out)
        for block in self.layer4:
            out = block.forward(out)

        return out @ self.fc

