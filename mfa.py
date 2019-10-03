import numpy as np
import torch
from torch.distributions import multinomial
import math
from matplotlib import pyplot as plt


class MFA(torch.nn.Module):
    def __init__(self, n_components, n_features, n_factors, isotropic=True, init_method='rnd_samples'):
        super(MFA, self).__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.n_factors = n_factors
        self.isotropic = isotropic
        self.init_method = init_method

        self.MU = torch.nn.Parameter(torch.zeros(n_components, n_features), requires_grad=False)
        self.A = torch.nn.Parameter(torch.zeros(n_components, n_features, n_factors), requires_grad=False)
        self.D = torch.nn.Parameter(torch.zeros(n_components, n_features), requires_grad=False)
        self.PI = torch.nn.Parameter(torch.ones(n_components)/float(n_components), requires_grad=False)
        # self.PI_logits = torch.nn.Parameter(torch.zeros(n_components))

    def fit(self, x):
        """
        EM Training
        :param x:
        :return:
        """
        pass

    def log_responsibilities(self, x):
        pass

    def sample(self, n, with_noise=True):
        K, d, l = self.A.shape
        # c_nums = multinomial.Multinomial(total_count=n, probs=self.PI).sample().long()
        c_nums = np.random.choice(K, n, p=self.PI.detach().numpy())
        z_l = torch.randn(n, l)
        z_d = torch.randn(n, d) if with_noise else torch.zeros(n, d)
        samples = torch.stack([self.A[c_nums[i]] @ z_l[i] + self.MU[c_nums[i]] + z_d[i] * self.D[c_nums[i]]
                               for i in range(n)])
        return samples, c_nums

    def per_component_log_likelihood(self, x):
        K, d, l = self.A.shape

        # Create some temporary matrices to simplify calculations...
        A = self.A
        AT = A.permute(0, 2, 1)
        iD = torch.pow(self.D, -2.0).view(K, d, 1)
        L = torch.eye(l).reshape(l, l, 1) + AT @ (iD*A)
        iL = torch.inverse(L)

        def per_component_md(i):
            x_c = (x - self.MU[i].reshape(1, d)).T  # shape = (d, n)
            m_d_1 = (iD[i] * x_c) - ((iD[i] * A[i]) @ iL[i]) @ (AT[i] @ (iD[i] * x_c))
            return torch.sum(x_c * m_d_1, dim=0)

        m_d = torch.stack([per_component_md(i) for i in range(K)])

        det_L = torch.log(torch.det(L))
        log_det_Sigma = det_L - torch.sum(torch.log(iD.reshape(K, d)), axis=1)
        log_prob_data_given_components = -0.5 * ((d*np.log(2.0*math.pi) + log_det_Sigma).reshape(K, 1) + m_d)
        # component_log_probs = (torch.log(torch.softmax(self.PI_logits)), [K, 1])
        return self.PI.reshape(K, 1) + log_prob_data_given_components

    def log_prob(self, x):
        return torch.logsumexp(self.per_component_log_likelihood(x), dim=0)

# Some unit testing...
if __name__ == '__main__':
    mfa = MFA(2, 2, 1)
    mfa.MU[0] = torch.FloatTensor([-1., 0.])
    mfa.MU[1] = torch.FloatTensor([1., 0.])
    mfa.A[0] = torch.FloatTensor([1.0/math.sqrt(2), 1.0/math.sqrt(2)]).reshape(2, 1)
    mfa.A[1] = torch.FloatTensor([0., 1.]).reshape(2, 1)
    mfa.D[0] = torch.FloatTensor([0.1, 0.1])
    mfa.D[1] = torch.FloatTensor([0.2, 0.2])


    samples, labels = mfa.sample(10)
    lls = mfa.per_component_log_likelihood(samples)
    print(labels)
    print(lls[0]-lls[1])

    # samples, _ = mfa.sample(1000)
    # samples = samples.numpy()
    # plt.plot(samples[:, 0], samples[:, 1], '.', alpha=0.5)
    # plt.show()
