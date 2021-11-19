from functools import reduce

import numpy as np
import numpy.typing as npt
import math
import tensorflow as tf

def is_zero(x: np.float_, y: np.float_) -> np.bool_:
    """
    x > y > 0
    """
    result: np.float_ = x / y - np.floor(x / y, dtype=np.float_)
    return result < 0.1


def lcm(xs: npt.NDArray[np.float_]) -> np.float_:
    return reduce(_lcm, xs)


def mod(x: np.float_, y: np.float_) -> np.float_:
    result: np.float_ = x - y * np.floor(x / y)
    return result


def _gcd(x: np.float_, y: np.float_) -> np.float_:
    """
    x > y
    """
    n = 0
    while y != 0 and not is_zero(x, y) and n < 10:
        x, y = y, mod(x, y)
        n += 1

    if is_zero(x, y):
        return y
    else:
        return x


def _lcm(x: np.float_, y: np.float_) -> np.float_:
    if x > y:
        return x * y / _gcd(x, y)
    else:
        return x * y / _gcd(y, x)

def minus_and_round(x):                             #dBを負に変換し、小数第5位までを切り捨て
    return math.floor(-x * 10 ** 4) / (10 ** 4)

def mean_percentage_squared_error(y_true,y_pred):   #平均二乗パーセント誤差
    percentage_error = (y_true - y_pred) / y_true
    loss = tf.math.reduce_mean(percentage_error**2)
    return loss