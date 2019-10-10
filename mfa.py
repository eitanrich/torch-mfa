import numpy as np
import torch
from torch.distributions import multinomial
import math
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler

#TODO: Rename to M PPCA and change D to be a scalar?

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

    def log_responsibilities(self, x):
        pass

    def sample(self, n, with_noise=True):
        K, d, l = self.A.shape
        # c_nums = multinomial.Multinomial(total_count=n, probs=self.PI).sample().long()
        c_nums = np.random.choice(K, n, p=self.PI.detach().cpu().numpy())
        z_l = torch.randn(n, l, device=self.A.device)
        z_d = torch.randn(n, d, device=self.A.device) if with_noise else torch.zeros(n, d, device=self.A.device)
        samples = torch.stack([self.A[c_nums[i]] @ z_l[i] + self.MU[c_nums[i]] + z_d[i] * self.D[c_nums[i]]
                               for i in range(n)])
        return samples, c_nums

    def per_component_log_likelihood(self, x):
        K, d, l = self.A.shape

        # Create some temporary matrices to simplify calculations...
        A = self.A
        AT = A.permute(0, 2, 1)
        iD = torch.pow(self.D, -2.0).view(K, d, 1)
        L = torch.eye(l, device=self.A.device).reshape(1, l, l) + AT @ (iD*A)
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

    def fit(self, x, max_iterations=20):
        """
        Estimates Maximum Likelihood MPPCA parameters for the provided data using EM
        :param x:
        :param max_iterations:
        :return:
        """
        K, d, l = self.A.shape
        N = x.shape[0]

        self._init_from_data(x, samples_per_component=(l+1)*2)

        def per_component_m_step(i):
            mu_i = torch.sum(r[:, [i]] * x, dim=0) / sum_r[i]
            s2_I = torch.pow(self.D[i, 0], 2.0) * torch.eye(l, device=x.device)
            inv_M_i = torch.inverse(self.A[i].T @ self.A[i] + s2_I)
            x_c = x - mu_i.reshape(1, d)
            SiAi = (1.0/sum_r[i]) * (r[:, [i]]*x_c).T @ (x_c @ self.A[i])
            invM_AT_Si_Ai = inv_M_i @ self.A[i].T @ SiAi
            A_i_new = SiAi @ torch.inverse(s2_I + invM_AT_Si_Ai)
            t1 = torch.trace(A_i_new.T @ (SiAi @ inv_M_i))
            trace_S_i = torch.sum(N/sum_r[i] * torch.mean(r[:, [i]]*x_c*x_c, dim=0))
            sigma_2_new = (trace_S_i - t1)/d
            return mu_i, A_i_new, torch.sqrt(sigma_2_new) * torch.ones_like(self.D[i])

        for it in range(max_iterations):
            r = self.responsibilities(x)
            sum_r = torch.sum(r, dim=0)
            if it%5 == 0:
                print('Iteration {}: log_likelihood = {}'.format(it, torch.mean(self.log_prob(x))))
            else:
                print('Iteration {}'.format(it))
            new_params = [torch.stack(t) for t in zip(*[per_component_m_step(i) for i in range(K)])]
            self.MU.data = new_params[0]
            self.A.data = new_params[1]
            self.D.data = new_params[2]

    def batch_fit(self, dataset: Dataset, batch_size=1000, max_iterations=20):
        """
        Estimates Maximum Likelihood MPPCA parameters for the provided dataset using batched-EM.
        Batching is performed on each step separately i.e. (E, E,.., E | M, M, ... M), (E, E,.., E | M, M, ... M), ...
        i.e. Iteration 1 E step on entire data, Iteration 1 M step on entire data, Iteration 2 E step on entire data...
        """
        K, d, l = self.A.shape

        # Initial guess
        init_samples_per_component = (l+1)*2
        init_keys = [key for i, key in enumerate(RandomSampler(dataset)) if i < init_samples_per_component*K]
        init_samples, _ = zip(*[dataset[key] for key in init_keys])
        self._init_from_data(torch.stack(init_samples), samples_per_component=init_samples_per_component)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for it in range(max_iterations):

            print('\nIteration {} / {}:'.format(it, max_iterations))
            # Step 1: Fetch all data and calculate and store all responsibilities, calculate mu
            mu_weighted_sum = torch.zeros(size=[K, d], dtype=torch.float64)
            all_r = []
            for batch_x, _ in loader:
                print('E', end='', flush=True)
                batch_r = self.responsibilities(batch_x)
                mu_weighted_sum += torch.stack([torch.sum(batch_r[:, [i]] * batch_x, dim=0).double() for i in range(K)])
                all_r.append(batch_r)
                # if len(all_r)==4:
                #     break
            all_r = torch.cat(all_r)
            print()

            # Update MU
            r_sum = torch.sum(all_r, dim=0).double()
            self.MU.data = (mu_weighted_sum / r_sum.reshape(-1, 1)).float()

            # Step 2 - Fetch all data again and Calculate all:
            # - SiAi - The empirical covariance matrices multiplied by the factors matrices
            # - Ri Xc^2 - The weighted empirical variance (for the noise variance calculation)
            SA = torch.zeros([K, d, l], dtype=torch.float64)
            RXc = torch.zeros(K, dtype=torch.float64)
            for batch_num, (batch_x, _) in enumerate(loader):
                print('M', end='', flush=True)
                batch_r = all_r[batch_num*batch_size: (batch_num+1)*batch_size]
                for i in range(K):
                    xc_i = batch_x-self.MU[i]
                    SA[i] += ((batch_r[:, [i]] * xc_i).T @ (xc_i @ self.A[i])).double()
                    RXc[i] += torch.sum(batch_r[:, [i]] * torch.pow(xc_i, 2.0)).double()
                # Alternative implementation:
                # SA += torch.stack([(batch_r[:, [i]]*(batch_x-self.MU[i])).T @
                #                           ((batch_x-self.MU[i]) @ self.A[i]) for i in range(K)])
                # RXc += torch.stack([batch_r[:, [i]]*torch.pow(batch_x-self.MU[i], 2.0)  for i in range(K)])
                # if batch_num == 3:
                #     break
            SA /= r_sum.reshape(-1, 1, 1)
            # Step 3 - Finalize the EM step
            s2_I = torch.pow(self.D[:, 0], 2.0).reshape(K, 1, 1) * torch.eye(l, device=self.MU.device).reshape(1, l, l)
            inv_M = torch.inverse((self.A.transpose(1, 2) @ self.A + s2_I).double())   # (K, l, l)
            invM_AT_S_A = inv_M @ self.A.double().transpose(1, 2) @ SA   # (K, l, l)
            self.A.data = (SA @ torch.inverse(s2_I.double() + invM_AT_S_A)).float()      # (K, d, l)
            t1 = torch.stack([torch.trace(self.A[i].double().T @ (SA[i] @ inv_M[i])) for i in range(K)])
            self.D.data = torch.sqrt((RXc / r_sum - t1)/d).float().reshape(-1, 1) * torch.ones_like(self.D)

    @staticmethod
    def _small_sample_ppca(x, n_factors):
        # See https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
        mu = torch.mean(x, dim=0)
        U, S, V = torch.svd(x - mu.reshape(1, -1))
        sigma_squared = torch.sum(torch.pow(S[n_factors:], 2.0))/((x.shape[0]-1) * (x.shape[1]-n_factors))
        A = V[:, :n_factors] * torch.sqrt((torch.pow(S[:n_factors], 2.0).reshape(1, n_factors)/(x.shape[0]-1) - sigma_squared))
        return mu, A, torch.sqrt(sigma_squared) * torch.ones(x.shape[1], device=x.device)

    def _init_from_data(self, x, samples_per_component):
        assert self.init_method == 'rnd_samples'

        K = self.n_components
        n = x.shape[0]
        l = self.n_factors
        m = samples_per_component
        o = np.random.choice(n, m*K, replace=False)
        assert n >= m*K
        params = [torch.stack(t) for t in zip(
            *[MFA._small_sample_ppca(x[o[i*m:(i+1)*m]], n_factors=l) for i in range(K)])]

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
    samples = samples.cpu().numpy()
    plt.plot(samples[:, 0], samples[:, 1], '.', alpha=0.5)
    plt.show()

