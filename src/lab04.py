import numpy as np
from helper import MathHelper as mh

import matplotlib.pyplot as plt

from MultivariateGaussian import MVG


def maximum_likelihood_estimate(XND: np.ndarray, X1D: np.ndarray) -> None:
    m_ML, C_ML = mh.compute_mu_and_sigma(XND)
    print("Mean vector:")
    print(m_ML)
    print("Covariance matrix:")
    print(C_ML)
    mvg = MVG(XND)
    ll = mvg.predict(XND)
    print("Log likelihood:")
    print(ll)

    m_ML, C_ML = mh.compute_mu_and_sigma(X1D)
    print("Mean vector:")
    print(m_ML)
    print("Covariance matrix:")
    print(C_ML)

    mvg = MVG(mu=m_ML, sigma=C_ML)
    XPlot = np.linspace(-8, 12, 1000)

    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    plt.plot(XPlot.ravel(), mvg.pdf_GAU_ND(mh.v_row(XPlot)))
    plt.show()

    ll = mvg.predict(X1D)
    print("Log likelihood:")
    print(ll)


def main() -> None:
    XPlot = np.linspace(-8, 12, 1000)
    mu = np.ones((1, 1)) * 1.0
    sigma = np.ones((1, 1)) * 2.0
    mvg = MVG(mu=mu, sigma=sigma)

    pdfSol = np.load("../labs/lab04/solution/llGAU.npy")
    pdfGau = mvg.logpdf_GAU_ND(mh.v_row(XPlot))
    print("First distance from solution (if ~=0 the solution is correct):")
    print(np.abs(pdfSol - pdfGau).max())

    XND: np.ndarray = np.load("../labs/lab04/solution/XND.npy")
    mu: np.ndarray = np.load("../labs/lab04/solution/muND.npy")
    C: np.ndarray = np.load("../labs/lab04/solution/CND.npy")
    mvg = MVG(mu=mu, sigma=C)

    pdfSol = np.load("../labs/lab04/solution/llND.npy")
    pdfGau = mvg.logpdf_GAU_ND(XND)

    print("Multivariate distance from solution (if ~=0 the solution is correct):")
    print(np.abs(pdfSol - pdfGau).max())

    # ML estimates - XND
    XND: np.ndarray = np.load("../labs/lab04/solution/XND.npy")
    X1D: np.ndarray = np.load("../labs/lab04/solution/X1D.npy")
    maximum_likelihood_estimate(XND, X1D)


if __name__ == "__main__":
    main()
