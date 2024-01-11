import numpy as np
import Corrfunc

import corracc

# n.o. threads Corrfunc should use
nthreads = 4
# n.o. (quasi) random point sets we want to use
Nr = 100000
Nq = Nr

# the test data supplied in Corrfunc, try  
#   wget https://github.com/manodeep/Corrfunc/raw/master/theory/tests/data/gals_Mr19.ff
# to download it from the Corrfunc repository
fname = "gals_Mr19.ff"
dx,dy,dz = Corrfunc .io.read_catalog(fname)
boxsize = 420.0

# radius bins (small and large scales)
bins = np.array([1,2,3,4,5,6,7, 70,75,80,85,90,95,100])


# LS estimator with exact DR and RR
#xiexact = corracc.xi_exact(dx,dy,dz, bins, boxsize, DD=DD, nthreads=nthreads)
#print("exact")
# precomputed xiexact (it takes some time) 
xiexact = np.array([8.69971919e+00, 3.23475756e+00, 1.85588725e+00,
                    1.28585641e+00, 9.53527462e-01, 7.42295116e-01, 
                    1.88524766e-02, 4.57868495e-03, 4.77627321e-03, 
                    4.52276456e-03, 4.87347602e-03, 4.60870626e-03, 
                    4.75552768e-03])

# LS standard 
xirnd = corracc.xiLSstandard(dx,dy,dz, bins, boxsize, Nr, nthreads=nthreads)
print("standard")

# LS qmc
xiqmc = corracc.xiLSqmc(dx,dy,dz, bins, boxsize, Nq, nthreads=nthreads)
print("qmc")

# the estimate for xi respecting periodic boundaries, we do not use it here
#xiper = corracc.xi_periodic(dx,dy,dz, bins, boxsize, nthreads=nthreads)

# some output
print("#  r      R    xiexact     xiLS        xiLSqmc")
for i in range(len(xiqmc)):
   print(f"{bins[i]: 6.1f} {bins[i+1]: 6.1f} ", end="")
   print(f"{xiexact[i]: 11.8f} {xirnd[i]: 11.8f} {xiqmc[i]: 11.8f}")
