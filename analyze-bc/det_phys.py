# useful physics calculations from moby2

import numpy as np

c = 2.99792458e8  #m/s
T_cmb = 2.7255    #K
k_pJ = 1.380649e-11   #pJ/K
h_pJ = 6.62607015e-22  #pJ s
pi = np.pi
GHz = 1e9


def blackbody_x(T, f):
    """
    Return dimensionless parameter
         x = h f / k T
    """
    return h_pJ / k_pJ * (f*GHz) / T

def rayleigh(T, f):
    """
    RJ spectral radiance, in (pW / m**2 / s / steradian) / (GHz).
    """
    return 2. * (f*GHz)**2 * k_pJ * T / c**2 * GHz

def blackbody(T, f):
    """
    Blackbody spectral radiance, in (pW / m**2 / s / steradian) / (GHz).
    """
    x = blackbody_x(T, f)
    return rayleigh(T, f) * x / (np.exp(x) - 1)

def spectrumToBrightness(S, f):
    """
    Given spectral radiance at f, returns the corresponding blackbody temperature.
    """
    x = np.log( 1. + ((c**2 / 2 / h_pJ / (f*GHz)**3) * S / GHz)**-1 )
    return h_pJ * (f*GHz) / k_pJ / x

def spectrumToRJ(S, f):
    """
    Given spectral radiance at f, returns the corresponding RJ temperature.
    """
    return 0.5 / (f*GHz/c)**2 / k_pJ * S / GHz

def blackbodyToRJ(T, f):
    return spectrumToRJ(blackbody(T, f), f)

def RJToBlackbody(T, f):
    return spectrumToBrightness(rayleigh(T, f), f)

def DCMB_factor(f):
    """
    Returns differential CMB temperature linearization factor X such
    that spectral radiance I is
      I = X * dT
    """
    x = blackbody_x(T_cmb, f)
    return GHz * k_pJ * \
           2 * (f*GHz / c)**2 * (x/2 / np.sinh(x/2))**2

def DCMB(dT, f):
    """
    Returns spectral radiance associated with source with temperature
    dT measured in differential CMB blackbody units.
    """
    return dT * DCMB_factor(f)

def spectrumToDCMB(S, f):
    """
    Given spectral radiance, return dT_CMB.
    """
    return S / DCMB_factor(f)

def RJToDCMB(f):
    """
    Returns the factor that converts RJ temperature to differential
    CMB temperature, at frequency f.
    """
    return spectrumToDCMB(rayleigh(1, f), f)

def microK_to_Jypersr(muK,f):
    """
    convert from muK units to Jy/sr based on https://arxiv.org/pdf/1303.5070.pdf & https://arxiv.org/pdf/2010.16405.pdf
    i.e. by taking a derivative of the Planck function

    args:
    muK: sensitivity [muK]
    f: frequency [Hz]

    output:
    sensitivity [Jy/sr]
    """

    kb=k_pJ*1.e-12
    hplanck=h_pJ*1.e-12


    x=hplanck*f/(kb*T_cmb)
    factor=2.*hplanck**2/(c**2*kb*(T_cmb*1.e6)*T_cmb)*1.e26
    Jypersr_muK=factor*(f)**4*np.exp(x)/(np.exp(x)-1.)**2

    return muK*Jypersr_muK

def Jypersr_to_microK(Jysr, f):
    """
    convert from Jy/sr units to muK

    args:
    Jysr: sensitivity [Jy/sr]
    f: frequency [Hz]

    output:
    sensitivity [muK]
    """

    Jysr_onemuK=microK_to_Jypersr(1.,f)

    return Jysr*1./Jysr_onemuK
