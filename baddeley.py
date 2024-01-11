# To calculate the volume integrals for the exact version
# of the Landay & Szalay estimator we need the area fraction
# in a cubic window. The funcions
#   c3,c2, a1,a2,a3, sphesfrac
# implement some of the (corrected) expressions from
#   A.J. Baddeley, R.A. Moyeed, C.V.Howard, A.Boyde,
#   Appl. Statist., 42, 641 (1993).
# This Python implementation follows closely the C-code in
# sphefrac.c (written by A.J. Baddeley) from the R package
# spatstat https://spatstat.org/ .
# Any errors in the python reimplementation are due to me.
# For details see
#   M. Kerscher, A&A 666, pp.  (2022)
#   https://arxiv.org/abs/2203.13288
import numpy as np

# this is C divided by pi (assumes a,b,c > 0)
def c3(a,b,c):
    if a*a +b*b + c*c >= 1.0:
        return 0.
    za = np.sqrt(1. - b*b - c*c)
    zb = np.sqrt(1. - a*a - c*c)
    zc = np.sqrt(1. - a*a - b*b)
    sum = np.arctan2(zb, a*c) + np.arctan2(za, b*c) + np.arctan2(zc, a*b)
    sum += -a*np.arctan2(zb, c) + a*np.arctan2(b, zc) 
    sum += -b*np.arctan2(za, c) + b*np.arctan2(a, zc) 
    sum += -c*np.arctan2(zb, a) + c*np.arctan2(b, za)
    return sum/np.pi - 1.


# this is C(a,b,0) divided by pi (assumes a,b, > 0)
def c2(a,b):
    z2 = 1. - a*a - b*b
    if z2 < 0:
        return 0.
    z = np.sqrt(z2)
    s = np.arctan2(z,a*b) - a*np.arctan2(z,b) - b*np.arctan2(z,a)
    return s/np.pi


# this is A1 divided by 4 pi r^2
def a1(t,r):
    if t>=r:
        return 0.
    return (1. - t/r)/2.


# this is A2 divided by 4 pi r^2
def a2(t1,t2,r):
    return c2(t1/r, t2/r)/2.


# this is A3 divided by 4 pi r^2
def a3(t1,t2,t3,r):
    return c3(t1/r, t2/r, t3/r)/4.


# see Baddeley et al, 1993 eq.(27)
def sphesfrac(x, r,L):
    p = x  # left lower [0 0 0]
    q = L - x
    
    sum=0.0
    for i in range(3):
        sum += a1(p[i],r) + a1(q[i],r)

    for i in range(3):
        for j in range(i+1,3):
            sum -= a2(p[i],p[j],r) + a2(p[i],q[j],r)
            sum -= a2(q[i],p[j],r) + a2(q[i],q[j],r)

    sum += a3(p[0],p[1],p[2],r)
    sum += a3(p[0],p[1],q[2],r) + a3(p[0],q[1],p[2],r) + a3(p[0],q[1],q[2],r)
    sum += a3(q[0],p[1],p[2],r) + a3(q[0],p[1],q[2],r) + a3(q[0],q[1],p[2],r)
    sum += a3(q[0],q[1],q[2],r)

    return 1.0-sum


# the fraction of the area inside
def area_fraction(x, r,L):
    return 4*np.pi*r*r * sphesfrac(x, r,L)


#  integrated isotropic set covariance int_r^R \overline{\gamma}_W(s) s^2 ds
def int_isotropic_setcov_cuboid(r,R, L):
    v = 1./3. * np.prod(L) * (R**3 - r**3)
    v -= 1./8. * (L[0]*L[1] + L[0]*L[2] + L[1]*L[2]) * (R**4 - r**4)
    v += 2./(15.*np.pi) * np.sum(L) * (R**5 - r**5)
    v -= 1./(24.*np.pi) * (R**6 - r**6)
    return v


# area fraction averaged over all points
def average_area_fraction(xn, r,L):
    a = 0.
    for q in xn:
        a += area_fraction(q, r,L)
    return a/len(xn)
