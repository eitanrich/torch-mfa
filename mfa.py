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

    # def fit(self, x):
    #     """
    #     EM Training
    #     :param x:
    #     :return:
    #     """
    #     pass

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
        return self.PI.reshape(1, K) + log_prob_data_given_components.T

    def log_prob(self, x):
        return torch.logsumexp(self.per_component_log_likelihood(x), dim=1)

    def log_responsibilities(self, x):
        comp_LLs = self.per_component_log_likelihood(x)
        return comp_LLs - torch.logsumexp(comp_LLs, dim=1).reshape(-1, 1)

    def responsibilities(self, x):
        return torch.exp(self.log_responsibilities(x))

    def fit(self, x, max_iterations=100):
        K, d, l = self.A.shape
        N = x.shape[0]

        self._init_from_data(x)

        def per_component_m_step(i):
            mu_i = torch.sum(r[:, [i]] * x, dim=0) / sum_r[i]
            s2_I = torch.pow(self.D[i, 0], 2.0) * torch.eye(l)
            inv_M_i = torch.inverse(self.A[i].T @ self.A[i] + s2_I)
            x_c = x - mu_i.reshape(1, d)
            SiAi = (1.0/sum_r[i]) * (r[:, [i]]*x_c).T @ (x_c @ self.A[i])
            invM_AT_Si_Ai = inv_M_i @ self.A[i].T @ SiAi
            A_i_new = SiAi @ torch.inverse(s2_I + invM_AT_Si_Ai)
            t1 = torch.trace(A_i_new.T @ (SiAi @ inv_M_i))
            trace_S_i = torch.sum(N/sum_r[i] * torch.mean(r[:, [i]]*x_c*x_c, dim=0))
            sigma_2_new = (trace_S_i - t1)/d
            return mu_i, A_i_new, torch.sqrt(sigma_2_new) * torch.ones(d)

        for it in range(max_iterations):
            r = self.responsibilities(x)
            sum_r = torch.sum(r, dim=0)
            print('Iteration', it)
            new_params = [torch.stack(t) for t in zip(*[per_component_m_step(i) for i in range(K)])]
            self.MU.data = new_params[0]
            self.A.data = new_params[1]
            self.D.data = new_params[2]

    @staticmethod
    def _small_sample_ppca(x, n_factors):
        # See https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
        mu = torch.mean(x, dim=0)
        U, S, V = torch.svd(x - mu.reshape(1, -1))
        sigma_squared = torch.sum(torch.pow(S[n_factors:], 2.0))/(x.shape[1]-n_factors)
        A = V[:, :n_factors] * torch.sqrt((torch.pow(S[:n_factors], 2.0).reshape(1, n_factors)/(x.shape[0]-1) - sigma_squared))
        return mu, A, torch.sqrt(sigma_squared) * torch.ones(x.shape[1])

    def _init_from_data(self, x):
        assert self.init_method == 'rnd_samples'

        K = self.n_components
        n = x.shape[0]
        l = self.n_factors
        m = l*2     # number of samples per component
        params = [torch.stack(t) for t in zip(
            *[MFA._small_sample_ppca(x[np.random.choice(n, size=m, replace=False)], n_factors=l) for i in range(K)])]

        self.MU.data = params[0]
        self.A.data = params[1]
        self.D.data = params[2]


# Some unit testing...
if __name__ == '__main__':
    mfa = MFA(2, 2, 1)
    mfa.MU[0] = torch.FloatTensor([-1., 0.])
    mfa.MU[1] = torch.FloatTensor([1., 0.])
    mfa.A[0] = torch.FloatTensor([1.0/math.sqrt(2), 1.0/math.sqrt(2)]).reshape(2, 1)
    mfa.A[1] = torch.FloatTensor([0., 1.]).reshape(2, 1)
    mfa.D[0] = torch.FloatTensor([0.1, 0.1])
    mfa.D[1] = torch.FloatTensor([0.2, 0.2])

    # samples, labels = mfa.sample(10)
    # log_probs = mfa.log_prob(samples)
    # lls = mfa.per_component_log_likelihood(samples)
    # r = mfa.responsibilities(samples)
    # print(samples.shape, lls.shape, r.shape)
    # print(log_probs)
    # print(labels)
    # print(lls[:, 0]-lls[:, 1])
    # print(r)

    samples, _ = mfa.sample(1000)
    # samples = samples.numpy()
    # plt.plot(samples[:, 0], samples[:, 1], '.', alpha=0.5)
    # plt.show()

    mfa2 = MFA(2, 2, 1)
    mfa2.fit(samples)

    samples, _ = mfa2.sample(1000)
    samples = samples.numpy()
    plt.plot(samples[:, 0], samples[:, 1], '.', alpha=0.5)
    plt.show()

