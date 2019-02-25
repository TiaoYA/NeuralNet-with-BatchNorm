import matplotlib.pyplot as plt
import numpy as np

def parse(file):
    with open(file, "rt") as f:
        lines = f.read().splitlines()
        N = len(lines)
        X = []
        Y = []
        for i in range(N):
            line = lines[i]
            ele = np.array(line.split(","))
            ele = map(float, ele)
            X.append(ele[:-1])
            Y.append(ele[-1])
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

def plot(x, y):
    x = x[1:]
    addNums = np.reshape(x,(28, 56))
    image1 = plt.imshow(addNums)
    plt.show()


if __name__ == "__main__":
    X, Y = parse("toy.txt")
    plot(X[-1], Y[-1])
    # we get 5, 8 and result 13.0
