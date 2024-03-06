import numpy as np
from scipy.linalg import polar

def RotationMatrix(v,w):
    ##following steps acc to whabhas algorithm
    ## M = Ux(N.T)x(X.T)xN
    ##compute square matrix A 
    A = np.matmul(v,w.T)
    ##decomposition of A into U and P
    U, P = polar(A)
    ##compute N and d from positive definate matrix P
    eigValues,N = np.linalg.eig(P)
    d = np.diag(eigValues)
    ###compute X depending on detU
    detU =  np.linalg.det(U)
    k = len(eigValues)
    epsilon = 0.1
    if detU > -1 - 0.1 and detU < -1 + 0.1:
        print("detU is -1")
        X = np.zeros((k,k))
        X[0:k-1, 0:k-1] = np.eye(k-1)
        X[-1,-1] = -1
    elif detU > 1 - 0.1 and detU < 1 + 0.1:
        print("detU is 1")
        X = np.eye(k) 
    else:  
        print("error U is not orthogonal")
    ##compute M using M = Ux(N.T)x(X.T)xN
    M = np.matmul(np.matmul(U,(N.T),(X.T)), N).T
    return M

def check_error(V,W,M,norm):
    ## check via l2 norm
    Vtransformed = np.dot(M,V)
    if norm == 'fro':
        error = np.linalg.norm(Vtransformed - W, ord='fro')  # Frobenius norm
    elif norm == 2:
        error = np.linalg.norm(Vtransformed - W, ord=2)  # 2-norm
    return error

##Q1 define matrix
V = np.array([[0.63,0.45,0.87],
              [0.03,0.40,0.86],
              [0.86,0.17,0.16],
              [0.29,0.79,0.04],
              [0.60,0.38,0.15],
              [0.99,0.43,0.76]])

W = np.array([[0.17,0.03,0.50],
              [0.26,0.37,0.15],
              [0.78,0.65,0.67],
              [0.83,0.96,0.82],
              [0.69,0.72,0.04],
              [0.64,0.77,0.34]])

M = RotationMatrix(V.T,W.T)
print(M)
##display errors in predicting each point in R3
errors = []
for i in range(V.shape[0]):
    ei = check_error(V[i,:].T,W[i,:].T,M,2)
    errors.append(ei)
print(f"Ecludian errors is estimating each point is {errors}")    
print(f"net minimized forbenius norm obtained = {check_error(V.T, W.T, M, 'fro')}")