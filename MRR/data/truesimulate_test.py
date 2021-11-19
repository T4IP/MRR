import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def simulate_transfer_function(
    wavelength: npt.NDArray[np.float64],
    L: npt.ArrayLike,
    K: npt.ArrayLike,
    alpha: float,
    eta: float,
    n_eff: float,
    n_g: float,
    center_wavelength: float,
) -> npt.NDArray[np.float64]:
    y: npt.NDArray[np.float64] = 20 * np.log10(np.abs(_D(L, K, alpha, wavelength, eta, n_eff, n_g, center_wavelength)))
    return y.reshape(y.size)


def _C(K_k: float, eta: float) -> npt.NDArray[np.float64]:
    C: npt.NDArray[np.float64] = (
        1
        / (-1j * eta * np.sqrt(K_k))
        * np.array([[1, -eta * np.sqrt(eta - K_k)], [np.sqrt(eta - K_k) * eta, -(eta ** 2)]])
    )
    return C


def _R(
    a_k: float,
    L_k: float,
    wavelength: npt.NDArray[np.float64],
    n_eff: float,
    n_g: float,
    center_wavelength: float,
) -> npt.NDArray[np.float64]:
    N_k = np.round(L_k * n_eff / center_wavelength)
    shifted_center_wavelength = L_k * n_eff / N_k
    x = (
        1j
        * np.pi
        * L_k
        * n_g
        * (wavelength - shifted_center_wavelength)
        / shifted_center_wavelength
        / shifted_center_wavelength
    )
    return np.array([[np.exp(x) / np.sqrt(a_k), 0], [0, np.exp(-x) * np.sqrt(a_k)]], dtype="object")


def _M(
    L: npt.ArrayLike,
    K: npt.ArrayLike,
    alpha: float,
    wavelength: npt.NDArray[np.float64],
    eta: float,
    n_eff: float,
    n_g: float,
    center_wavelength: float,
) -> npt.NDArray[np.float64]:
    L_: npt.NDArray[np.float64] = np.array(L)[::-1]
    K_: npt.NDArray[np.float64] = np.array(K)[::-1]
    a: npt.NDArray[np.float64] = np.exp(-alpha * L_)
    product = np.identity(2)
    for K_k, a_k, L_k in zip(K_[:-1], a, L_):
        product = np.dot(product, _C(K_k, eta))
        product = np.dot(product, _R(a_k, L_k, wavelength, n_eff, n_g, center_wavelength))
    product = np.dot(product, _C(K_[-1], eta))
    return product


def _D(
    L: npt.ArrayLike,
    K: npt.ArrayLike,
    alpha: float,
    wavelength: npt.NDArray[np.float64],
    eta: float,
    n_eff: float,
    n_g: float,
    center_wavelength: float,
) -> npt.NDArray[np.float64]:
    D: npt.NDArray[np.float64] = 1 / _M(L, K, alpha, wavelength, eta, n_eff, n_g, center_wavelength)[0, 0]
    return D

L=np.array([82.4e-6,82.4e-6,55.0e-6,82.4e-6,55.0e-6,82.4e-6,55.0e-6,55.0e-6])
K=np.array([0.33, 0.11, 0.02, 0.05, 0.14, 0.73, 0.59, 0.05, 0.31])
#L=np.array([82.4e-6,55.0e-6])
#K=np.array([0.7, 0.08, 0.47])
#data.print_parameters()
plt.plot(np.arange(1520,1560.01,0.01),simulate_transfer_function(np.arange(1520e-9,1560e-9,0.01e-9),L,K,alpha=53,eta=0.996,n_eff=2.2,n_g=4.4,center_wavelength=1550e-9),label = "original")
#K=np.array([0.6935,0.7833,0.5055])
#plt.plot(np.arange(1520,1560.01,0.01),simulate_transfer_function(np.arange(1520e-9,1560e-9,0.01e-9),L,K,alpha=53,eta=0.996,n_eff=2.2,n_g=4.4,center_wavelength=1550e-9),label = "predict")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Drop Port Power (dB)")
plt.legend(bbox_to_anchor=(1,0),loc="lower right")
plt.show()

