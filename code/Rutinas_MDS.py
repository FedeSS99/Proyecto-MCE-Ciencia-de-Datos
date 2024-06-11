import numpy as np

class ClustTimeMDS:
    def __init__(self, dissimilarity:np.ndarray) -> None:
        self.__dis2 = np.copy(dissimilarity)**2.0

        self.N = self.__dis2.shape[0]
        self.__H = np.eye(N = self.N) - (1/self.N)*np.full((self.N, self.N), 1.0)


    def __ComputeB(self):
        self.__B = -0.5 * (self.__H @ self.__dis2 @ self.__H)

    def __GetNthEigenValue(self) -> np.float32:
        B_eigvals = np.linalg.eigvals(self.__B)
        B_eigvals = B_eigvals[B_eigvals > 0]

        return B_eigvals.min()

    def __GetEuclideanDistances(self):
        MinEigVal = self.__GetNthEigenValue()

        self.__EuclidDist = self.__dis2 - 2.0 * MinEigVal * (np.full((self.N, self.N), 1.0) - np.eye(N = self.N))

    def GetSolution(self, num_dim:int):
        self.__ComputeB()
        self.__GetEuclideanDistances()
        B_euclid = - 0.5 * (self.__H @ self.__EuclidDist @ self.__H)

        EigVals_B_euclid, EigVecs_B_euclid = np.linalg.eig(B_euclid)
        EigVals_B_euclid = EigVals_B_euclid[EigVals_B_euclid > 0]
        if num_dim <= EigVals_B_euclid.size:
            EigVals_red_B_euclid = EigVals_B_euclid[:num_dim]
            EigVecs_red_B_euclid = EigVecs_B_euclid[:,:num_dim]

            self.VarExplained = 100 * EigVals_red_B_euclid.sum()/EigVals_B_euclid.sum()
            print(f"{self.VarExplained} of explained variance with {num_dim} components")

            self.Xc = np.sqrt(EigVals_red_B_euclid) * EigVecs_red_B_euclid