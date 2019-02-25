import numpy as np
import matplotlib.pyplot as plt

def sigmoid(vec):
    return 1 / (1 + np.exp(-vec))

def softmax(vec):
    return np.exp(vec) / sum(np.exp(vec))

def preAct(W, b, X):
    return b + np.dot(W, X)

# assuming sigmoid activision
def hiddenAct(vec):
    return sigmoid(vec)

def outAct(vec):
    return softmax(vec)

def shuffleData(x, y):
    assert len(x) == len(y)
    s = np.random.permutation(len(x))
    return x[s], y[s]

def crossEntropy(predict, label):
    iY = np.transpose(oneHotKey(label))
    k = -1 * np.sum(iY * np.log(predict)) / float(len(label))
    return k

def meanCE(predict, label):
    mi = np.argmax(predict, axis=0)
    c = np.equal(mi, label)
    return 1 - sum(c) / float(len(label))

# transform y to onehotkeyenocoding
def oneHotKey(y):
    K = 19
    N = len(y)
    iY = np.zeros([N, K])
    iY[np.arange(N), y.astype(int)] = 1
    return iY

def sigdiv(vec):
    return (1 - vec) * vec

def plotCEE(trainE, valE):
    X = [i for i in range(len(trainE))]
    training, = plt.plot(X, trainE, color='r', label = "Average Training Cross-Entropy Error")
    validation, = plt.plot(X, valE, color='b', label = "Average Validation Cross-Entropy Error")
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Error')
    plt.legend()
    plt.show()

def plotMCE(tE, valE):
    X = [i for i in range(len(tE))]
    training, = plt.plot(X, tE, color='r', label = "Mean Training Classification Error")
    validation, = plt.plot(X, valE, color = 'b', label = "Mean Validation Classification Error")
    plt.xlabel('Epochs')
    plt.ylabel('Mean Classification Error')
    plt.legend()
    plt.show()

def visualP(W):
    fig, axis = plt.subplots(nrows=10, ncols=10)
    for col in range(10):
        for row in range(10):
            image = np.reshape(W[col*10+row],(28,56))
            axis[row, col].imshow(image)
            axis[row, col].set_axis_off();
            plt.subplots_adjust(hspace=0.1)
    plt.show()

'''
def finite_diff(x, y, theta):
    epsilon = 1e-5
    k = len(theta)
    grad = np.zeros(k)
    for m in range(k)
        d = np.zeros(k)
'''

