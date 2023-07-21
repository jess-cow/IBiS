import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy
import time
import pymaster as nmt


def gen_gaussian_map(v, pp, deg):
    vp = v[pp]
    mu = np.dot(v, vp)
    G = np.exp(-(1-mu)/np.radians(deg/2.355)**2)
    return G

def int_bispectrum(mp, deg, mask=None, fil='Gauss',
                   filename=None, save=False, alpha_weight=0):
    '''calculate integrated bispectrum estimator.
    Parameters:
    mp: map 
    deg:degree size of patch 
    '''
    if filename is not None: 
        if os.path.isfile(filename):
            f = np.load(filename)
            return f['IB'], f['map_mean'], f['cls'], f['fpatch_map'], f['fsky_map']

    npix = len(mp) #number of pixels
    nside = hp.npix2nside(npix) #nside of map
    ipix = np.arange(hp.nside2npix(nside))
    vecmap = np.array(hp.pix2vec(nside, ipix)).T
    theta_patch = np.radians(deg)
    cls = []
    map_mean = np.zeros(npix)
    fpatch_map = np.zeros(npix)
    fsky_map = np.zeros(npix)
    if mask is None:
        mask = np.ones(npix)

    for ip in range(npix):
        v = hp.pix2vec(nside, ip) #vector location of pixel
        if fil=='Tophat':
            idisc = hp.query_disc(nside, v, theta_patch) #Returns pixels whose centers lie within the disk defined by *vec* and *radius*
            w = np.zeros(npix)
            w[idisc] = 1. #set to 1 where we have patch, 0 elsewhere, i.e. tophat filter
        elif fil=='Gauss':
            #gaussian filter
            w = gen_gaussian_map(vecmap, ip, deg)

        fpatch = np.mean((w*mask)**2)
        fsky = np.mean(w*mask)
        Npatch = npix*fsky  # number of pixels in patch (won't be integer due if gauss filter)

        map_mult = w * mp  # masked map times filter, 
        cl_ip = hp.anafast(map_mult, iter=0)/fpatch  # cl in patch
        map_mean[ip] = np.sum(map_mult) / Npatch #mean density of patch
        cls.append(cl_ip)
        fpatch_map[ip] = fpatch
        fsky_map[ip] = fsky
    cls = np.array(cls)  # [npix, n_ells]
    if alpha_weight == 0:
        weight_map = np.ones(npix)
    else:
        weight_map = fsky_map**alpha_weight
    IB = np.average(map_mean[:, None]*cls[:, :], axis=0,
                    weights=weight_map)

    if filename is not None:
        np.savez(filename, map_mean=map_mean, IB=IB, cls=cls,
                 fpatch_map=fpatch_map, fsky_map=fsky_map)
    return IB, map_mean, cls, fpatch_map, fsky_map
