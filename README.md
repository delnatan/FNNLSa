# FNNLSa
Python port of FNNLSa algorithm by Rasmus Bro. Directly taken from his MATLAB file exchange:

<http://www.mathworks.com/matlabcentral/fileexchange/3388-nnls-and-constrained-regression?focused=5051382&tab=function>

This translation requires NumPy.

Note that for FNNLS, the symmetric matrix $A^TA$ and matrix-vector product $A^T b$ is used as arguments, so `FNNLSa(AtA, Atb)`. The NNLS implementation from `scipy.optimize` uses `nnls(A,b)` instead. 

Running `%timeit` magic with IPython notebook with a toy problem (see ipynb file in this repository):

__scipy.optimize NNLS__

`41 ms ± 358 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)`

__FNNLSa__

`9.73 ms ± 51.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)`

For a transformation matrix (600 x 1000)



