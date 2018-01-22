# FNNLSa
Python port of FNNLSa algorithm by Rasmus Bro. Directly taken from his MATLAB file exchange:

<http://www.mathworks.com/matlabcentral/fileexchange/3388-nnls-and-constrained-regression?focused=5051382&tab=function>

This translation requires NumPy.

Note that for FNNLS, the symmetric matrix $A^TA$ and matrix-vector product $A^T b$ is used as arguments, so `FNNLSs(AtA, Atb)`. The NNLS implementation from `scipy.optimize` uses `nnls(A,b)` instead. 


### Example


