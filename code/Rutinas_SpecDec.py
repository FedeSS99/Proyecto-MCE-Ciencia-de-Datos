import numpy as np

class ClustTimeSD:
    def __init__(self, Similarity:np.ndarray, eps:float, max_iters:int, sig0:float, alpha:float) -> None:
        self.__eps: float = eps
        self.__max_iters: int = max_iters
        self.__sig: float = sig0
        self.__alpha: float = alpha

        self.__Similarity = np.copy(Similarity)
        
        self.__GetAffiinityMatrix()
        self.__GetOptimalNumberOfClusters()

    def __GetAffiinityMatrix(self):
        self.Affinity = np.exp( - 0.5 * ((self.__Similarity/self.__sig) ** 2.0))
        np.fill_diagonal(self.Affinity, 0.0)

    def __GetOptimalNumberOfClusters(self):
        U = np.diag( self.Affinity.sum(axis = 1) )
        self.__Affinityp = np.linalg.pinv(U) @ self.Affinity

        Affinity_eigenvalues = np.linalg.eigvals(self.__Affinityp)

        self.OptimalNumClusters = np.isclose(Affinity_eigenvalues, np.ones(Affinity_eigenvalues.size)).sum()
        print(f"Optimal number of clusters = {self.OptimalNumClusters}")

    def __ComputeG(self):
        self.__GetAffiinityMatrix()
        Q = np.linalg.eig(self.Affinity)[1]

        G = 0.0
        for k in range(self.OptimalNumClusters):
            q_outer_product = np.outer(Q[:,k], Q[:,k])
            Partial_S = ((self.__Similarity/self.__sig)**2.0) * self.Affinity / self.__sig
            G += (q_outer_product * Partial_S).sum()

        return G, Q[:,:self.OptimalNumClusters]
    
    def SolveForOptimalQ(self):
        for i in range(self.__max_iters):
            G, Q_actual = self.__ComputeG()

            absG = abs(G)
            if absG < self.__eps:
                print(f"G is lower than the given precision with {i + 1} iterations")
                break
            else:
                self.__sig += self.__alpha * G

        if i == self.__max_iters - 1 and absG >= self.__eps:
            print(f"Max iterations reached with {G=:}")

        self.FoundQ = Q_actual.real
