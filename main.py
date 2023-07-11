import numpy as np


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [0], [0], [1]])

INPUT_DIM = X.shape[1]
HIDDEN_DIM = 5
lr = 0.1
epochs = 1000

# initialize weights
# input -> hidden
w1 = np.random.randn(INPUT_DIM, HIDDEN_DIM)
b1 = np.random.randn(1, HIDDEN_DIM)
w2 = np.random.randn(HIDDEN_DIM, 1)
b2 = np.random.randn(1, 1)

# Activation function
sigmoid = lambda x: 1 / (1 + np.exp(-x))
# Loss function
loss = lambda y, y_hat: np.mean(np.square(y - y_hat))

# Forward pass
def forward(x):
    # input -> hidden
    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)
    # hidden -> output
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    return a1, a2
    
# Backward pass
for epoch in range(epochs): 
    a1, a2 = forward(X)
    # output -> hidden
    delta2 = (a2 - Y) * a2 * (1 - a2)
    dw2 = np.dot(a1.T, delta2) 
    db2 = np.sum(delta2, axis=0, keepdims=True)
    # hidden -> input
    delta1 = np.dot(delta2, w2.T) * a1 * (1 - a1)
    dw1 = np.dot(X.T, delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)
    # update weights
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2
    # print loss
    if epoch % 100 == 0:
        print(f"Loss at epoch {epoch}: {loss(Y, a2)}")
