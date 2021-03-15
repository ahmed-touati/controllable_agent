import numpy as np
import torch


class RadialBasisFunction2D(object):
    def __init__(self, size, dim, sigma, cuda=False):
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.size = size
        self.dim = dim
        self.sigma = sigma

        xlist = np.linspace(0, size, dim)
        ylist = np.linspace(0, size, dim)
        XX, YY = np.meshgrid(xlist, ylist)

        self.XX = XX
        self.YY = YY

        self.x_mu = FloatTensor(XX.flatten())
        self.y_mu = FloatTensor(YY.flatten())

    def transform(self, A):
        distance = (A[:, 0].reshape(-1, 1) - self.x_mu[None])**2 + (A[:, 1].reshape(-1, 1) - self.y_mu[None])**2
        # X_mu = np.broadcast_to(self.x_mu, (A.shape[0], self.dim**2))
        # Y_mu = np.broadcast_to(self.y_mu, (A.shape[0], self.dim**2))
        # X = np.broadcast_to(A[:,0].reshape(-1,1), (A.shape[0], self.dim**2))
        # Y = np.broadcast_to(A[:,1].reshape(-1,1), (A.shape[0], self.dim**2))
        #
        # distance = ((X - X_mu)**2+((Y - Y_mu)**2))
        weights = torch.exp(-distance/(2 * (self.sigma**2)))
        return weights / weights.sum(axis=1, keepdims=True)

    def inverse_transform(self, A):
        index = np.argmax(A, axis=1)
        result = []
        for idx in index:
            i, j = self._1d_index_to_2d_index(idx)
            result.append([self.XX[i][j], self.YY[i][j]])

        return np.array(result)

    def _1d_index_to_2d_index(self, index):
        i = index // self.dim
        j = index % self.dim

        return i, j


if __name__ == '__main__':
    featurizer = RadialBasisFunction2D(10, 11, 1)
    A = torch.FloatTensor(np.array([[5, 5], [3, 6]]))
    print(featurizer.inverse_transform(featurizer.transform(A)))


