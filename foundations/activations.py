import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        #z = 1 / (1+pow(np.exp(1),-z))
        #return np.round(z, 5)
        return np.round(1 / (1 + np.exp(-z)), 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        #if np.all(z) < 0:
        #    return 0
        #a = []
        #for i in z:
        #    a.append(float(max(0,i)))
        #return a
        return np.maximum(0, z)
        
