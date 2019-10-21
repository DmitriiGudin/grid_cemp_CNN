from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py


def plot_spectrum (f, N, filename, min_w="def", max_w="def"): 
    w = np.arange(3000, 10000+1, 1)
    s = f["/spectrum"][N]
    T = f["/T_EFF"][N][0]
    G = f["/LOG_G"][N][0]
    Z = f["/FE_H"][N][0]
    C = f["/C_FE"][N][0]
    if min_w == "def":
        min_w = min(w)
    if max_w == "def":
        max_w = max(w)
    min_w_index = np.where(w==min_w)[0][0]
    max_w_index = np.where(w==max_w)[0][0]
    w = w[min_w_index:max_w_index+1]
    s = s[min_w_index:max_w_index+1]
    plt.clf()
    plt.title ("Teff = " + str(T) + ";   log(g) = " + str(G) + ";   [Fe/H] = " + str(Z) + ";   [C/Fe] = " + str(C), size=24)
    plt.xlabel ("Wavelength (A)", size=24)
    plt.ylabel ("Flux", size=24)
    plt.xlim (min(w), max(w))
    plt.ylim (min(s), max(s))
    plt.tick_params(labelsize=18)
    plt.gca().set_yscale("log")
    plt.plot (w, s, linewidth=3, color='black')
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig(filename, dpi=100)
    plt.close()


if __name__ == '__main__':
    f = h5py.File("spectra_raw.hdf5", "r")
    w = np.arange(3000, 10000+1, 1)

    #plot_spectrum (f, 87467, "spectrum_87467_cu.png", min_w=3000, max_w=3200)
    #plot_spectrum (f, 87467, "spectrum_87467.png", min_w=3000, max_w=10000)
    
    #plot_spectrum (f, 23043, "spectrum_23043.png", min_w=3000, max_w=10000)
  
    #plot_spectrum (f, 40000, "spectrum_40000.png", min_w=3000, max_w=10000)

    #plot_spectrum (f, 787, "spectrum_787.png", min_w=3000, max_w=10000)

    #plot_spectrum (f, 6390, "spectrum_6390.png", min_w=3000, max_w=10000)

    #plot_spectrum (f, 58533, "spectrum_58533.png", min_w=3000, max_w=10000)

    plot_spectrum (f, 54050, "spectrum_54050.png", min_w=3000, max_w=10000)

    plot_spectrum (f, 15129, "spectrum_15129.png", min_w=3000, max_w=10000)

    plot_spectrum (f, 35169, "spectrum_35169.png", min_w=3000, max_w=10000)

    plot_spectrum (f, 72220, "spectrum_72220.png", min_w=3000, max_w=10000)
