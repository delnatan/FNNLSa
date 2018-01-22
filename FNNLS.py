# FNNLSa implementation from Rasmus Bro (1997) fnnls.m (MATLAB file exchange)
# translated to Python
# Daniel Elnatan
# Jan 22, 2018
from numpy import zeros, arange, int64, float64, sum, argmax, nonzero, diag,\
    min, abs, newaxis, finfo
from scipy.linalg import solve
nu = newaxis

# machine epsilon
eps = finfo(float64).eps

def any(a):
    # assuming a vector, a
    larger_than_zero = sum(a > 0)
    if larger_than_zero:
        return True
    else:
        return False

def find_nonzero(a):
    # returns indices of nonzero elements in a
    return nonzero(a)[0]

def FNNLSa(XtX, Xty, tol=eps):
    # default tolerance is machine epsilon, otherwise you can put in argument
    M,N = XtX.shape

    # initialize passive set, P. Indices where coefficient is >0
    P   = zeros(N, dtype=int64)
    # and active set. Indices where coefficient is <=0
    Z   = arange(N)
    # working active set
    ZZ = arange(N)
    # initial solution vector, x
    x   = zeros(N, dtype=float64)
    # weight vector
    w   = Xty - XtX @ x
    # iteration counts and parameter
    it = 0 
    itmax = 30 * N

    # MAIN LOOP
    # continue as long as there are indices within the active set Z
    # or elements in inner loop active set is larger than 'tolerance'
    while any(Z) and any(w[ZZ] > tol):
        t = argmax(w[ZZ]) # find largest weight
        t = ZZ[t] 
        P[t] = t # move to passive set
        Z[t] = 0 # remove from active set
        PP   = find_nonzero(P)
        ZZ   = find_nonzero(Z)
        NZZ  = ZZ.size
        # compute trial solution, s
        s     = zeros(N, dtype=float64)
        if len(PP) == 1:
            s[PP] = Xty[PP]/XtX[PP,PP]
        else:
            s[PP] = solve(XtX[PP,PP[:,nu]], Xty[PP])
        s[ZZ] = 0.0 # set active coefficients to 0 
    
        while any(s[PP] <= tol) and it < itmax:
            it = it + 1
            QQ = find_nonzero( (s <= tol) * P )
            alpha = min( x[QQ] / (x[QQ] - s[QQ]) ) 
            x     = x + alpha * (s - x)
            ij    = find_nonzero( (abs(x)<tol) * (P!=0) )
            Z[ij] = ij
            P[ij] = 0
            PP    = find_nonzero(P)
            ZZ    = find_nonzero(Z)
            NZZ   = ZZ.size
            if len(PP) == 1:
                s[PP] = Xty[PP]/XtX[PP,PP]
            else:
                s[PP] = solve(XtX[PP,PP[:,nu]], Xty[PP])
            s[ZZ] = 0.0
        # assign current solution, s, to x
        x = s
        # recompute weights
        w = Xty - XtX @ x

    return x, w