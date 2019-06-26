# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, M Zieba
#  2019
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np

with open('naive.pkl', mode='rb') as file_:
    mXd = pkl.load(file_)

def simplify(x, threshold):
    x[x >= threshold] = 1
    x[x < threshold] = 0
    return x


def probabilities(X, p_x_l_y):
    p_x_1_y_rev = 1 - p_x_l_y
    X_rev = 1 - X
    res = []  # N*M
    for i in range(X.shape[0]):
        present = np.power(p_x_l_y, X[i])  # MxD ^ 1xD = MxD
        absent = np.power(p_x_1_y_rev, X_rev[i])  # MxD ^ 1xD = MxD
        distribution_bin = np.prod(np.multiply(present, absent),
                                   axis=1)  # prod(MxD) * 1xM = 1xM,  p(x|y') * p(y') max. likelihood
        # każda cecha niezależna wiec iloczyn
        sum = np.sum(distribution_bin)
        if (sum == 0):
            sum = 1
        res.append(distribution_bin / sum)
    return np.array(res)

def pickLabel(x):
    labels = np.argmax(x, axis=1)
    return labels


def noiseReduction(x):
    x = np.reshape(x,(36,36))
    left =7
    right=35
    north=7
    south=35

    for a in range(28,35):
        noise = 0
        for b in range(35):
            if (x[a][b]-x[a][b+1]):
                noise+=1
        if(noise>=35):
            south = a
            north = south-27-1
            break
    for c in range(28,35):
        noise = 0
        for d in range(35):
            if (x[d][c] - x[d+1][c]):
                noise += 1
        if (noise >= 35):
            right = c
            left = right - 27-1
            break
    newX = x[north:south,left:right]
    return np.reshape(newX,(1,28**2))

def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 9} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """

    x_without_noise = np.zeros((x.shape[0], 28 ** 2))
    for k in range(x.shape[0]):
        x_without_noise[k] = noiseReduction(x[k])
    x_without_noise = simplify(x_without_noise, 0.15)
    probs = probabilities(x_without_noise,mXd)
    predictions = pickLabel(probs)
    final = np.reshape(predictions,(predictions.shape[0],1))

    return final
    pass
