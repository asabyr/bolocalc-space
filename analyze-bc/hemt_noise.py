import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from astropy.modeling import models
import astropy.constants as c
import astropy.units as u

class hemt:
    
    def __init__(self, band_edges, eta):
        # takes in a tuple of band edges and returns the HEMT sensitivity for each band
        self.band_edges = np.array(band_edges) * u.GHz
        self.freqs = np.arange(0.5, 150, 0.01) * u.GHz
        self.eta = eta
        T_cmb = self.cmb()
        T_fgs = self.foregrounds()
        T_hemt = self.amplifier()
        T_sys = T_cmb + T_fgs + T_hemt
        self.sens, self.T_sys, self.T_sky = [], [], []
        for f_low,f_high in self.band_edges:
            idx = np.where( (self.freqs >= f_low) & (self.freqs < f_high) )[0]
            tsys = np.mean((T_cmb+T_fgs+T_hemt)[idx])
            tsky = np.mean((T_cmb+T_fgs)[idx])
            self.T_sys.append(tsys)
            self.T_sky.append(tsky)
            self.sens.append(self.dicke_sens(tsys, f_low, f_high))
        self.popt = self.opt_pow()

    def cmb(self):
        bb = models.BlackBody(temperature=2.726*u.K, scale=1*u.J/(u.m ** 2 * u.s * u.Hz * u.sr))
        spec_rad = bb(self.freqs)
        T_cmb = spec_rad.to(u.K, equivalencies=u.brightness_temperature(self.freqs))
        return T_cmb
    
    def foregrounds(self):
        fg_labels = ['Freqs', 'Synch', 'Free-free', 'AME', 'CIB', 'Dust', 'CO', 'Total']  
        data = np.loadtxt('foregrounds/foregrounds_muK.txt')
        fgs_uk = {}
        d = np.where(data[:,0] <= 200*1e9)
        for j,col in enumerate(data.T):
            fgs_uk[fg_labels[j]] = {}
            fgs_uk[fg_labels[j]] = col[d]
        f = interp1d(fgs_uk['Freqs']*1e-9, fgs_uk['Total'])
        T_fgs = f(self.freqs) * 1e-6 * u.K
        return T_fgs

    def amplifier(self):
        hemt_noise = [1.1,1.1,1.3,1.5,1.5,2,3,5,6.5,8,10,8,9,12,10,10,15,17,20,22,27,30]
        hemt_freqs = [1,2,3,4,6,8,10,16,20,25,28,30,40,45,50,59,69,77,80,90,100,116]
        z = np.polyfit(hemt_freqs, hemt_noise, 2)
        hemt_fit = np.poly1d(z)
        T_hemt = hemt_fit(self.freqs.value) * u.K
        return T_hemt

    def dicke_sens(self, tsys, f_low, f_high):
        bw = f_high - f_low
        sens = np.mean(tsys) / np.sqrt(bw*self.eta)
        return sens.to(u.uK / (u.Hz)**0.5)

    def opt_pow(self):
        return np.nan
