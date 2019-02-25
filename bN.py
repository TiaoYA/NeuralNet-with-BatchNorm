import numpy as np
import matplotlib.pyplot as plt
import parse
import sys
from helper import *

# K: label classes, output, 19
# L: # of (hidden) layers
# D_i: # of (hidden) units of ith (hidden) layer
# M: dim of input instance X_i
# N: total # of instance

# input: layer: list of # of nodes in ith layer
# output: adding gamma and beta
def batch_init_para(layers):
    para = []
    for i in range(len(layers) - 1):
        thisLayer = []
        bias = np.zeros((layers[i + 1],1))
        seed = np.sqrt(6) / np.sqrt(layers[i] + layers[i + 1])
        weight = 2 * np.random.random_sample((layers[i + 1], layers[i])) * seed - seed
        gamma = 1
        beta = 0
        thisLayer.append(bias)
        thisLayer.append(weight)
        thisLayer.append(gamma)
        thisLayer.append(beta)
        para.append(thisLayer)
    return para

# code adapted from kevinzakka's blog
# forward function for batching parameters
def batch_norm(x, gamma, beta, batchSize=32, epsilon=1e-5):
    mu = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    x_norm = (x-mu)/np.sqrt(var + epsilon)
    out = gamma*x_norm + beta
    return mu, var, x_norm, out

def forward(para, x, bS = 32, epsilon=1e-5):
    L = len(para)
    # in form of H0: M * N, a1: D1 * M, H1, a2, H2, aout, Hout
    cache = []
    # H0
    cache.append([np.transpose(x)])
    for i in range(L - 1):
        a = preAct(para[i][1], para[i][0], cache[-1][0])
        gamma = para[i][-2]
        beta = para[i][-1]
        mu, var, x_n, out = batch_norm(a, gamma, beta, bS, epsilon)
        h = sigmoid(out)
        cache.append([h] + [a] + [out] + [mu] + [var] + [x_n])
    # Hout
    aOut = preAct(para[-1][1], para[-1][0], cache[-1][0])
    outG = para[-1][-2]
    outB = para[-1][-1]
    outMu, outV, outXN, outY = batch_norm(aOut, outG, outB, bS, epsilon)
    hOut = softmax(aOut)
    cache.append([hOut] + [aOut] + [outY] + [outMu] + [outV] + [outXN])
    return cache

# from kevinzakka's github post
def batch_backward(dout, cache, gamma):
    N = dout.shape[0]
    x_mu, inv_var, x_hat = cache
    dxhat = dout * gamma
    dx = (1. / N) * inv_var * (N*dxhat - np.sum(dxhat, axis=1, keepdims=True) - x_hat*np.sum(x_hat*dxhat, axis=1, keepdims=True)) 
    dbeta = np.sum(dout, axis=1, keepdims=True)
    dgamma = np.sum(x_hat*dout, axis=1, keepdims=True)
    return dx, dgamma, dbeta

def backward(cache, para, x, y):
    N = float(x.shape[0]) # number of instances
    L = len(cache)
    grad = []
    # outer
    iY = np.transpose(oneHotKey(y))
    dH = - iY / cache[-1][0]
    dA = -1 * (iY - cache[-1][0])
    grad.append([dA] + [dH])
    for i in range(L - 1, 0, -1):
        dY, dGamma, dBeta = batch_backward(grad[-1][0], cache[i][3:], para[i-1][-2])
        # cache[i-1][0]: H_{i-1}
        h = np.transpose(cache[i - 1][0])
        dB = np.sum(grad[-1][0], axis=1, keepdims=True) / N
        dW = np.dot(grad[-1][0], h) / N
        # i-1th layer
        dH = np.dot(np.transpose(para[i - 1][1]), grad[-1][0])
        if i != 1: dA = dH * sigdiv(cache[i-1][0])
        else: dA = dH
        grad[-1].append(dB)
        grad[-1].append(dW)
        grad[-1].append(dY)
        grad[-1].append(dGamma)
        grad[-1].append(dBeta)
        grad.append([dA] + [dH])
    # in reverse order
    return (grad[::-1])[1:]

def batch_onestep(x, y, para, prevG, j, e, lamb=0, momentum=0, l_rate=0.1):
    cache = forward(para, x)
    thisJ = j
    thisJ += crossEntropy(cache[-1][0], y)
    e += meanCE(cache[-1][0], y)
    grad = backward(cache, para, x, y)
    # add momentum
    if len(prevG) != 0 and momentum != 0:
        for i in range(len(grad)):
            for j in range(len(grad[i])):
                grad[i][j] = (1 - momentum) * grad[i][j] + momentum * prevG[i][j]
    # update
    for i in range(len(para)):
        # format b_i, w_i, gamma_i, beta_i
        # format of grad: dA, dH, dB, dW, d_gamma, d_beta
        para[i][0] -= grad[i][2] * l_rate
        para[i][1] -= grad[i][3] * l_rate
        para[i][2] -= grad[i][-2] * l_rate
        para[i][3] -= grad[i][-1] * l_rate
        if lamb != 0:
            factor = 1 - 10 ** (-lamb)
            para[i][1] *= factor
            para[i][2] *= factor
            para[i][3] *= factor
    return grad, para, thisJ, e

def test(testx, testy, parameters):
    cache = forward(parameters, testx)
    c = crossEntropy(cache[-1][0], testy)
    m = meanCE(cache[-1][0], testy)
    return c, m

def train(tx, ty,vx,vy, epoche=150, b_size=32, lamb=0, momentum=0, l_rate=0.1, hidLayer=1, hidnode1=100, hidnode2=0):
    if hidLayer == 1: 
        assert(hidnode2 == 0)
        build_init_para = [28*56, hidnode1, 19]
    else:
        build_init_para = [28*56, hidnode1, hidnode2, 19]
    para = batch_init_para(build_init_para)
    grad = []
    tN = tx.shape[0]
    numBatch = tN / b_size
    e_t = []
    e_v = []
    meanErr_t = []
    meanErr_v = []
    for i in range(epoche):
        t_j = 0
        t_e = 0
        v_j = 0
        v_e = 0
        tx, ty = shuffleData(tx, ty)
        for b in range(numBatch):
            xB = tx[b*b_size: (b+1)*b_size]
            yB = ty[b*b_size : (b+1) *b_size]
            pG = grad
            vC = forward(para, vx)
            v_e += meanCE(vC[-1][0], vy)
            v_j += crossEntropy(vC[-1][0], vy)
            grad, para, t_j, t_e = batch_onestep(xB, yB, para, pG, t_j, t_e,lamb, momentum, l_rate)
        e_t.append(t_j/float(numBatch))
        e_v.append(v_j/float(numBatch))
        meanErr_t.append(t_e/float(numBatch))
        meanErr_v.append(v_e/float(numBatch))
    return para, e_t, e_v, meanErr_t, meanErr_v

if __name__ == "__main__":
    epoch = int(sys.argv[1])
    b_size = int(sys.argv[2])
    decay = int(sys.argv[3])
    mom = float(sys.argv[4])
    l_r = float(sys.argv[5])
    h_l = int(sys.argv[6])
    hn1 = int(sys.argv[7])
    hn2 = int(sys.argv[8])

    tx, ty = parse.parse("train.txt")
    vx, vy = parse.parse("val.txt")
    #testx, testy=parse.parse("toy.txt")
    para, e_t, e_v, mean_t, mean_v = train(tx, ty, vx, vy, epoch, b_size, decay, mom, l_r, h_l, hn1, hn2)
    print("trainCross: " + str(e_t[-1]) + "\n")
    print("trainErr: " + str(mean_t[-1]) + "\n")
    print("valCross: " + str(e_v[-1]) + "\n")
    print("valErr: " + str(mean_v[-1]) + "\n")
    testx, testy = parse.parse("test.txt")
    a, b = test(testx, testy, para)
    print("testCrossEntropy: " + str(a) + "\n")
    print("testMeanClassificationErr: " + str(b) + "\n")
    plotCEE(e_t, e_v)
    plotMCE(mean_t, mean_v)
    visualP(para[0][1])
