# Created by Dmitrii Gudin (U Notre Dame).
# Modifies the code from spectrum.py by Devin Whitten.

from __future__ import division
import spectrum

class Spectrum (spectrum.Spectrum):
    
    # Creates a Spectrum object with the given wavelength and flux arrays.
    def __init__(self, wavelength, flux):
        self.wavelength = wavelength
        self.flux = flux
    
    # Returns the continuum for the spectrum.
    def get_continuum(self):
        self.generate_segments()
        self.assess_segment_variation()
        self.define_cont_points()
        self.spline_continuum()
        return self.continuum

    # Returns the normalized spectrum.
    def get_flux_norm(self):
        self.generate_segments()
        self.assess_segment_variation()
        self.define_cont_points()
        self.spline_continuum()
        self.normalize()
        return self.flux_norm
