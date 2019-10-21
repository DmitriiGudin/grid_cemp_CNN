#Author: Devin Whitten
#Date: Nov 12, 2016
# This is will serve as the interface for the normalization function.
# So just defining some functions in here.


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from segment import Segment

################################
#Spectrum Class Definition
################################
class Spectrum():
    def __init__(self, fitsfile):
        #### Use the fits file to generate all necessary info for the spectrum
        self.fits = fitsfile
        self.wavelength = np.power(10. , (self.fits[0].header['CRVAL1'] + np.arange(0, self.fits[0].header['NAXIS1'])*self.fits[0].header['CD1_1']))
        self.flux = self.fits[0].data[0]

        #### Defined in generate_segments
        self.segments = None
        ####
        self.mad_global = None

        return

    def generate_segments(self, bins=25):
        self.segments = [Segment(wl, flux) for wl, flux in zip(np.array_split(self.wavelength, bins), np.array_split(self.flux, bins))]
        ### Need to handle the end points!!!!!!
        self.segments[0].is_edge("left")
        self.segments[-1].is_edge('right')

        [segment.get_statistics() for segment in self.segments]
        return


    def assess_segment_variation(self):
        ### Precondition: self.segments must exist, generate_segments must have previously been run
        self.mad_array = np.array([segment.mad for segment in self.segments], dtype=float)
        self.mad_global = np.median(self.mad_array)
        self.mad_min = min(self.mad_array)
        self.mad_max = max(self.mad_array)
        self.mad_range = self.mad_max - self.mad_min

        self.mad_normal = np.divide(self.mad_array - self.mad_min, self.mad_range)

    def define_cont_points(self):
        ### just runs define_cont_point in the Segment class, which boosts median
        ### by a scale estimate normalized to the distribution of mads from
        ### each segment

        ### Precondition: must run assess_segment_variation
        [segment.define_cont_point(self.mad_min, self.mad_range) for segment in self.segments]


    ### Accessors
    def get_segment_midpoints(self):
        self.midpoints = [segment.midpoint for segment in self.segments]
        ## add endpoints

        return np.array(self.midpoints, dtype=np.float)

    def get_segment_continuum(self):
        self.fluxpoints = [segment.continuum_point for segment in self.segments]
        ## add fluxpoints

        return np.array(self.fluxpoints, dtype=np.float)



    def spline_continuum(self, k=2.0, s=1.0):
        ## Precondition:  define_cont_points has been run, segment.continuum_point exists

        ### Create the spline interpolation

        tck = interp.splrep(self.get_segment_midpoints(), self.get_segment_continuum())

        self.continuum = interp.splev(self.wavelength, tck)


    def normalize(self):
        self.flux_norm = np.divide(self.flux, self.continuum)




    def poly_normalize(self, nlow=3.0, nhigh=3.0, boost=0.05, order=4, Regions=[]):
        return




    def spline_normalize(self, BINS=25):
        ### Trying to do this without any unneccesary parameters

        ### divide the spectrum into equal bins

        return
