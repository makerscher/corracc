import numpy as np
from scipy.stats import qmc
import scipy.integrate as integrate
import Corrfunc
import baddeley


def xiLSstandard(dx,dy,dz, bins, boxsize, Nr,rng=None, DD=None, nthreads=1):
    '''
    Landy & Szalay estimator for xi using random points 

    Parameters
    ----------
    dx : numpy.ndarray
        the x-coordinates of the data points
    dy : numpy.ndarray
        the y-coordinates of the data points
    dz : numpy.ndarray
        the z-coordinates of the datapoints
    bins : numpy.ndarray
        list of ordered bin boundaries for the radii.
    boxsize : float
        sidelength of the cubic box.
    Nr : int
        number of random points used.
    rng : numpy.random._generator.Generator, optional
        Supply one if you want to provide a random number generator in a
        definite state. The default is None.
    DD : numpy.ndarray, optional
        A special data structure from Corrfunc holding the data-data
        pair-counts. The default is None.
    nthreads : int, optional
        number of thread used by Corrfunc. The default is 1.

    Returns
    -------
    xirnd : numpy.ndarray
        the standard LS estimate for xi.

    '''
    if rng is None:
        rng = np.random.default_rng(1234567)
    if DD is None :
        DD = Corrfunc.theory.DD(autocorr=True,nthreads=nthreads,periodic=False, 
                              binfile=bins,boxsize=boxsize,
                              X1=dx, Y1=dy, Z1=dz)
    # random points
    rx = rng.uniform(0, boxsize, Nr)
    ry = rng.uniform(0, boxsize, Nr)
    rz = rng.uniform(0, boxsize, Nr)
    #
    DR = Corrfunc.theory.DD(autocorr=False,nthreads=nthreads,periodic=False, 
                            binfile=bins,boxsize=boxsize,
                            X1=dx,Y1=dy,Z1=dz, X2=rx,Y2=ry,Z2=rz)
    RR = Corrfunc.theory.DD(autocorr=True,nthreads=nthreads,periodic=False, 
                            binfile=bins,boxsize=boxsize,
                            X1=rx,Y1=ry,Z1=rz)
    # combine
    N = len(dx)
    xirnd = Corrfunc.utils.convert_3d_counts_to_cf(N,N,Nr,Nr, DD,DR,DR,RR)
    return xirnd



def xiLSqmc(dx,dy,dz, bins, boxsize, Nq,qmc6d=None, DD=None,nthreads=1):
    '''
    Landy & Szalay type estimator for xi using a randomized Haltom sequence

    Parameters
    ----------
    dx : numpy.ndarray
        the x-coordinates of the data points
    dy : numpy.ndarray
        the y-coordinates of the data points
    dz : numpy.ndarray
        the z-coordinates of the datapoints
    bins : numpy.ndarray
        list of ordered bin boundaries for the radii.
    boxsize : float
        sidelength of the cubic box.
    Nr : int
        number of random points used.
    qmc6d : scipy.stats._qmc.Halton, optional
        Supply a sequence generator if you want to provide a generator in a
        definite state. The default is None.
    DD : numpy.ndarray, optional
        A special data structure from Corrfunc holding the data-data
        pair-counts. The default is None.
    nthreads : int, optional
        number of thread used by Corrfunc. The default is 1.

    Returns
    -------
    xiqmc : numpy.ndarray
        the quasi MC LS type estimate for xi.

    '''
    if qmc6d is None:
        qmc6d = qmc.Halton(d=6, seed=1234567)
    if DD is None:
        DD = Corrfunc.theory.DD(autocorr=True,nthreads=nthreads,periodic=False, 
                                binfile=bins,boxsize=boxsize,
                                X1=dx, Y1=dy, Z1=dz)
    # two 3d point sets from a 6d randomized Halton sequence
    rn6d = qmc6d.random(Nq)*boxsize
    qx = rn6d[:,0]; qy = rn6d[:,1]; qz = rn6d[:,2]
    sx = rn6d[:,3]; sy = rn6d[:,4]; sz = rn6d[:,5]
    #
    DQ = Corrfunc.theory.DD(autocorr=False,nthreads=nthreads,periodic=False, 
                            binfile=bins,boxsize=boxsize,
                            X1=dx,Y1=dy,Z1=dz, X2=qx,Y2=qy,Z2=qz)
    QQ = Corrfunc.theory.DD(autocorr=False,nthreads=nthreads,periodic=False, 
                            binfile=bins,boxsize=boxsize,
                            X1=qx,Y1=qy,Z1=qz, X2=sx,Y2=sy,Z2=sz)
    # combine
    N = len(dx)
    xiqmc = Corrfunc.utils.convert_3d_counts_to_cf(N,N,Nq,Nq, DD,DQ,DQ,QQ)
    return xiqmc


def xi_exact(dx,dy,dz, bins, boxsize, DD=None,nthreads=1):
    '''
    Landy & Szalay estimator for xi using exact DR and RR

    Parameters
    ----------
    dx : numpy.ndarray
        the x-coordinates of the data points
    dy : numpy.ndarray
        the y-coordinates of the data points
    dz : numpy.ndarray
        the z-coordinates of the datapoints
    bins : numpy.ndarray
        list of ordered bin boundaries for the radii.
    boxsize : float
        sidelength of the cubic box.
    DD : numpy.ndarray, optional
        A special data structure from Corrfunc holding the data-data
        pair-counts. The default is None.
    nthreads : int, optional
        number of thread used by Corrfunc. The default is 1.

    Returns
    -------
    xi : numpy.ndarray
        the exact LS estimate for xi.

    '''
    # cubic box, points etc
    L = np.array([boxsize,boxsize,boxsize])
    pts = np.array([dx,dy,dz]).T
    ### data-data paircounts (from Corrfunc)
    if DD is None:
       DD = Corrfunc.theory.DD(autocorr=True,nthreads=nthreads,periodic=False, 
                               binfile=bins,boxsize=boxsize,
                               X1=dx,Y1=dy,Z1=dz)
    DD = DD['npairs']/(len(dx)*len(dx))
    ### RR exact
    # calculate the integrated isotropic set covariance for a box
    intisc = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        r = bins[i]
        R = bins[i+1]
        intisc[i] = baddeley.int_isotropic_setcov_cuboid(r,R, L) 
    # RR = 4 pi / |w|^2 * intisc
    RR = intisc * 4.*np.pi/(np.prod(L)**2)
    ### DR exact
    avafrac = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        r = bins[i]
        R = bins[i+1]
        resquad = integrate.quad(
           lambda q: baddeley.average_area_fraction(pts,q,L), r,R)
        # to control the accuracy of quad use e.g. epsabs=0., epsrel=1.e-10
        avafrac[i] = resquad[0]
    # DR = average_area_fraction / |W|
    DR = avafrac / np.prod(L)
    # combine and return LS estimator
    return (DD - 2.*DR)/RR + 1.


def xi_periodic(dx,dy,dz, bins, boxsize, nthreads=1):
    '''
    Estimate of xi from points in a box with periodic boundaries

    Parameters
    ----------
    dx : numpy.ndarray
        the x-coordinates of the data points
    dy : numpy.ndarray
        the y-coordinates of the data points
    dz : numpy.ndarray
        the z-coordinates of the datapoints
    bins : numpy.ndarray
        list of ordered bin boundaries for the radius.
    boxsize : float
        sidelength of the cubic box.
    nthreads : int, optional
        number of thread used by Corrfunc. The default is 1.

    Returns
    -------
    xi : numpy.ndarray
        estimate for xi assuming periodic boundaries.
        
    '''
    # data-data paircounts using Corrfunc with periodic boundaries
    DD = Corrfunc.theory.DD(autocorr=True,nthreads=nthreads,periodic=True, 
                            binfile=bins,boxsize=boxsize,
                            X1=dx,Y1=dy,Z1=dz)
    xi = DD['npairs']/(len(dx)**2)
    # from paircounts to xi in periodic box
    for i in range(len(xi)):
        r = bins[i]
        R = bins[i+1]
        # xi = |W|/(4 pi/3 (R^3 - r^3) ) * DD - 1
        xi[i] *= np.power(boxsize,3.)/(4.*np.pi/3.*
                                       (np.power(R,3.)-np.power(r,3.)))
        xi[i] -= 1.
    return xi

