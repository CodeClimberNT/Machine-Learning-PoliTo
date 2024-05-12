import numpy as np
from helper import MathHelper as mh

import matplotlib.pyplot as plt

from MultivariateGaussian import MVG


def main() -> None:

    XPlot = np.linspace(-8, 12, 1000)
    mu = np.ones((1, 1)) * 1.0
    sigma = np.ones((1, 1)) * 2.0
    mvg = MVG(mu, sigma)

    pdfSol = np.load("labs/lab04/solution/llGAU.npy")
    pdfGau = mvg.logpdf_GAU_ND(mh.v_row(XPlot))
    print("First solution (if 0.0 the solution is correct):")
    print(np.abs(pdfSol - pdfGau).max())

    XND: np.ndarray = np.load("labs/lab04/solution/XND.npy")
    mu: np.ndarray = np.load("labs/lab04/solution/muND.npy")
    C: np.ndarray = np.load("labs/lab04/solution/CND.npy")
    mvg = MVG(mu, C)

    pdfSol = np.load("labs/lab04/solution/llND.npy")
    pdfGau = mvg.logpdf_GAU_ND(XND)

    print(np.abs(pdfSol - pdfGau).max())

if __name__ == "__main__":
    main()
