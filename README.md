# corracc

## Description

This repo contains an example showing how one can calculate the two-point correlation function more accurately than with random points at no extra computational cost.

The basic idea is to use a low discrepance sequence instead of random points. 
We use randomized Halton sequences as provide by [SciPy](https://docs.scipy.org/doc/scipy/reference/stats.qmc.html)
and for the pair-counts we use [Corrfunc](https://github.com/manodeep/Corrfunc).
A brief explanation why such an approach improves the accuracy with no extra cost can be found in [corracc.pdf](https://github.com/makerscher/corracc/blob/main/corracc.pdf). See also [Kerscher 2022](https://arxiv.org/abs/2203.13288) for more details.

## Implementation and Example

In `corracc.py` the implementation of the standard Landy & Szalay estimator is given, followed by the estimator using a low discrepancy sequence. Both are using the pair-counts from Corrfunc. 
In `xi.py` the usage of these functions is illustrated with the test data set [gals_Mr19ff](https://github.com/manodeep/Corrfunc/raw/master/theory/tests/data/gals_Mr19.ff) from the Corrfunc repository.

In `corracc.py` also the implementation of the exact Landay & Szalay estimator and of an estimator for periodic boundary conditions is given. 
For the exact estimator we need the geometric functions from `baddeley.py`. The implementation closely follows  the C-code in sphefrac.c from the R package [spatstat](https://spatstat.org/) (written by A.J. Baddeley).
