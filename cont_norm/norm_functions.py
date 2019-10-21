#Author: Devin Whitten
#Date: Nov 12, 2016
## For now just the plotting functions


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
plt.ion()

plt.style.use("ggplot")

def plot(spec):
    fig, ax = plt.subplots(2,1)

    ax[0].plot(spec.wavelength, spec.flux)
    ax[0].set_xlabel(r"Wavelength $\AA$")
    ax[0].set_ylabel(r"Flux")

    ax[0].scatter(spec.get_segment_midpoints(), spec.get_segment_continuum(), s =10, zorder=3, color="black")

    ### Continuum
    ax[0].plot(spec.wavelength, spec.continuum, zorder=3, color="black")

    ### Normalization
    ax[1].plot(spec.wavelength, spec.flux_norm)

    plt.show()
    input("Press any key to continue")
    plt.close()





### APPEND NORMALIZED FLUX TO FITS
def update_fits(spec, fits):
    ### Precondition: spec needs flux_norm define
    ### we'll add the continuum to the fits file as well.

    fits[0].header['ARRAY5'] = "FLUX_NORM"
    fits[0].header['ARRAY6'] = "CONTINUUM"
    fits[0].data = np.vstack((fits[0].data, spec.flux_norm, spec.continuum))

    return fits
