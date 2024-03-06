import numpy as np
from ..nlforces.nonlinear_force import InstantaneousForce

class GenPolyForce(InstantaneousForce):
    """
    Cubic and Quadratic Polynomial Nonlinearity (Used for geometric nonlinearity)

    Parameters
    ----------
    Q : (Nnl,n) numpy.ndarray
        Transformation matrix from system DOFs (n) to nonlinear DOFs (Nnl)
    T : 
        Transformation matrix from local nonlinear forces to global 
        nonlinear forces, n x Nnl
    Emat: () numpy.ndarray
        Matrix to transform modal coordinates into modal nonlinear force
    qq : FILL IN THE REST OF THIS
        Matrix corresponding to exponnent of modal coordinates
        

    """
    
    def __init__(self, Q, T, Emat, qq):
        self.Q = Q
        self.T = T
        self.Emat = Emat
        self.qq = qq
        
    
    def force(self, X):
        
        unl = self.Q @ X
        
        fnl = self.Emat @ np.prod(unl ** self.qq, axis=1)
        
        F = self.T @ fnl
        
        dFdX =  self.T @ np.matmul(self.Emat , ((np.prod(unl.T ** self.qq, axis=1).reshape(-1, 1) \
                           / (unl.T ** self.qq + np.finfo(float).eps)) \
                   * (self.qq * (unl.T ** np.maximum(self.qq - 1, 0))))) @ self.Q
        
        return F, dFdX
    
    def local_force_history(self, unlt, unltdot):
        
        ## This function is modified to take multiple unlt rows from aft
                
        # #old
        #ft =  np.matmul(self.Emat , np.prod(np.power(unlt.T[:, np.newaxis, :],  self.qq), axis=2))
        ft=[]
        dfdu=[]
        
            
        for k_row in range(unlt.shape[0]):
             
             u1=unlt[k_row,:]
             ftcol =  self.Emat @ np.prod(u1 ** self.qq, axis=1)
             ft.append(ftcol)
             
             
             dfduarr= np.matmul(self.Emat, ((np.prod(u1 ** self.qq, axis=1).reshape(-1, 1) \
                  / (u1.T ** self.qq + np.finfo(float).eps)) \
                    * (self.qq * (u1.T ** np.maximum(self.qq - 1, 0)))))
             dfdu.append(dfduarr)
                  
               
        ft = np.hstack(ft)
        dfdu = np.stack(dfdu, axis=2)
        dfdud = np.zeros_like(unlt)

        return ft, dfdu, dfdud
    
    
