import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from astropy.modeling import models
import astropy.constants as c
import astropy.units as u
import os
from det_phys import RJToDCMB, Jypersr_to_microK

this_dir=os.path.dirname(os.path.abspath(__file__))

class hemt:

    def __init__(self, band_edges, eta):
        # takes in a tuple of band edges and returns the HEMT sensitivity for each band
        self.band_edges = np.array(band_edges) * u.GHz
        self.band_centers = (self.band_edges[:,1] + self.band_edges[:,0])/2
        self.freqs = np.arange(0.5, 250, 0.01) * u.GHz
        self.eta = eta
        T_cmb = self.cmb()
        T_fgs = self.foregrounds()
        T_hemt = self.amplifier()
        T_sys = T_cmb + T_fgs + T_hemt
        T_3ql = self.ideal_amp()
        self.sens, self.sens_3ql, self.T_sys, self.T_sys_3ql, self.T_sky, self.T_amp, self.T_amp_3ql = [], [], [], [], [], [], []
        for f_low,f_high in self.band_edges:
            idx = np.where( (self.freqs >= f_low) & (self.freqs < f_high) )[0]
            rj_to_cmb = RJToDCMB((f_low.value + f_high.value)/2)
            tsys = np.mean((T_cmb+T_fgs+T_hemt)[idx])
            tsys_3ql = np.mean((T_cmb+T_fgs+T_3ql)[idx])
            tsky = np.mean((T_cmb+T_fgs)[idx])
            tamp = np.mean(T_hemt[idx])
            t3ql = np.mean(T_3ql[idx])
            self.T_sys.append(tsys)
            self.T_sys_3ql.append(tsys_3ql)
            self.T_sky.append(tsky)
            self.T_amp.append(tamp.value)
            self.T_amp_3ql.append(t3ql.value)
            self.sens.append(rj_to_cmb * self.dicke_sens(tsys, f_low, f_high))
            self.sens_3ql.append(rj_to_cmb * self.dicke_sens(tsys_3ql, f_low, f_high))
        self.popt = self.opt_pow()
        self.T_sky = [i.value for i in self.T_sky]
        self.T_sys = [i.value for i in self.T_sys]
        self.T_sys_3ql = [i.value for i in self.T_sys_3ql]
        self.sens = [i.value for i in self.sens]
        self.sens_3ql = [i.value for i in self.sens_3ql]

    def cmb(self):
        bb = models.BlackBody(temperature=2.7255*u.K, scale=1*u.J/(u.m ** 2 * u.s * u.Hz * u.sr))
        spec_rad = bb(self.freqs)
        spec_rad_Jypersr=spec_rad*1.e26

        spec_rad_Jypersr_value=spec_rad_Jypersr.value
        freqs_value=self.freqs.value*1.e9

        T_cmb_muK=Jypersr_to_microK(spec_rad_Jypersr_value, freqs_value)
        T_cmb=T_cmb_muK*1e-6*u.K

        return T_cmb

    def foregrounds(self):
        fg_labels = ['Freqs', 'Synch', 'Free-free', 'AME', 'CIB', 'Dust', 'CO', 'Total']
        data = np.loadtxt(this_dir+'/foregrounds/foregrounds_muK.txt')
        fgs_uk = {}
        d = np.where(data[:,0] <= 300*1e9)
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

    def ideal_amp(self):
        h = 6.626e-34 * 1e9 # J / GHz
        k = 1.38e-23
        nu = self.freqs.value
        t_3ql = 3*h*nu/k #/ np.log(2)
        f = interp1d(nu, t_3ql, bounds_error=False, fill_value='extrapolate')
        return f(self.freqs) * u.K

    def dicke_sens(self, tsys, f_low, f_high):
        bw = f_high - f_low
        sens = np.sqrt(2.0)*np.mean(tsys) / np.sqrt(self.eta * bw)
        return sens.to(u.uK / (u.Hz)**0.5)

    def opt_pow(self):
        return np.nan
