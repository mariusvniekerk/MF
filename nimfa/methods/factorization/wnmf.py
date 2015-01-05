"""
###################################
Nmf (``methods.factorization.nmf``)
###################################

**Weighted Nonnegative Matrix Factorization (NMF)

"""


from nimfa.models import *
from nimfa.utils.linalg import *
from nimfa.methods.factorization.nmf import Nmf


class Wnmf(Nmf):


    def __init__(self, **params):
        self.name = "wnmf"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        nmf_std.Nmf_std.__init__(self, params)
        self.weights = params['options'].pop('weights')
        self.set_params()

    def euclidean_update(self):
        """Update basis and mixture matrix based on Euclidean distance multiplicative update rules.

        """

        self.H = multiply(
            self.H, elop(dot(self.W.T, multiply(self.V,self.weights)), dot(self.W.T, multiply(self.weights,dot(self.W, self.H))), div))

        self.W = multiply(
            self.W, elop(dot(multiply(self.V,self.weights), self.H.T), dot(multiply(dot(self.W,self.H),self.weights), self.H.T), div))

    def fro_objective(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate."""
        R = self.V - dot(self.W, self.H)
        return multiply(R, R).sum()