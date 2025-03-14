import torch
import torch.nn as nn
import torch.nn.functional as F


class CAAMIL(nn.Module):
    def __init__(self, L=2048, D=128, K=1, c_dim=5):
        super(CAAMIL, self).__init__()
        self.L = L   # input feature dim
        self.D = D    # hidden dim
        self.K = K      # num of classes
        self.c_dim = c_dim  # c dim

        self.cP_attn = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K))

        self.cP_clsf = nn.Sequential(
            nn.Linear(self.L, self.K),
            nn.Sigmoid()
        )

        self.P_attn = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K))

        self.P_clsf = nn.Sequential(
            nn.Linear(self.L, self.K),
            nn.Sigmoid()
        )


    def forward(self, xs):
        # x = (N,2048), cl_x = (N', 2048), km = (K, 2048), min_cl_p_count = (int)
        x, cl_x, km, min_cl_p_count = xs

        if x is not None:
            pH = x.squeeze(0)  # {h_1, ..., h_N}   (N,L)
            pA = self.P_attn(pH)  # N x c_dim
            pA = torch.transpose(pA, 1, 0)  # c_dim x N
            pA = F.softmax(pA, dim=1)  # softmax over N

            p_aggr = torch.mm(pA, pH)  # c_dim x L
            pY_prob = self.P_clsf(p_aggr)
            pY_hat = torch.ge(pY_prob, 0.5).float()

        if cl_x is not None:
            cpH = cl_x.squeeze(0)   # {h_1, ..., h_K}   (N,L)

        if km is None:
            cpA = self.cP_attn(cpH)  # N x c_dim
            cpA = torch.transpose(cpA, 1, 0)  # c_dim x N
            new_cpA = torch.clone(cpA)
            print(cpA.shape)

            if len(new_cpA[0])%min_cl_p_count:
                raise IOError('Error: len(new_cpA[0])%min_cl_p_count')

            cl_k = int(len(new_cpA[0])/min_cl_p_count)

            cA = []
            for _k in range(cl_k):
                cpA_mean = cpA[:, _k * min_cl_p_count:_k * min_cl_p_count + min_cl_p_count].mean()

                new_cpA[:, _k*min_cl_p_count:_k*min_cl_p_count + min_cl_p_count] = cpA_mean
                cA.append(cpA_mean)
            cA = torch.Tensor(cA).unsqueeze(0)

            new_cpA = F.softmax(new_cpA, dim=1)  # softmax over N
            cA = F.softmax(cA, dim=1)  # softmax over N

        else:

            cH = km.squeeze(0)
            cA = self.cP_attn(cH)  # N x c_dim
            cA = torch.transpose(cA, 1, 0)  # c_dim x N

            if isinstance(min_cl_p_count, list):
                new_cpA = []
                for _i, _cl in enumerate(min_cl_p_count):
                    new_cpA.append(cA.transpose(0,1)[_i].repeat(1,_cl))
                new_cpA = torch.cat(new_cpA,1)
            else:
                new_cpA = cA.repeat(min_cl_p_count, 1).transpose(0, 1).reshape(1, -1)

            new_cpA = F.softmax(new_cpA, dim=1)  # softmax over N
            cA = F.softmax(cA, dim=1)  # softmax over N

        p_aggr = torch.mm(new_cpA, cpH)  # c_dim x L
        cpY_prob = self.cP_clsf(p_aggr)
        cpY_hat = torch.ge(cpY_prob, 0.5).float()

        if x is not None:
            return (pY_prob, pY_hat, pA), (cpY_prob, cpY_hat, new_cpA), (None, None, cA)
        else:
            return (None, None, None), (cpY_prob, cpY_hat, new_cpA), (None, None, cA)


    def get_loss(self, pY_prob, cY_prob, Y, l=0.5):
        pY_prob = torch.clamp(pY_prob, min=1e-3, max=1. - 1e-3)
        cY_prob = torch.clamp(cY_prob, min=1e-3, max=1. - 1e-3)
        p_neg_log_likelihood = -1. * (
                    Y * torch.log(pY_prob) + (1. - Y) * torch.log(1. - pY_prob))  # negative log bernoulli
        c_neg_log_likelihood = -1. * (
                    Y * torch.log(cY_prob) + (1. - Y) * torch.log(1. - cY_prob))  # negative log bernoulli

        tot_loss = l*p_neg_log_likelihood + (1-l)*c_neg_log_likelihood
        return tot_loss


class AttnMIL(nn.Module):
    def __init__(self, L=2048, D=128, K=1):
        super(AttnMIL, self).__init__()
        self.L = L    # input feature dim
        self.D = D    # hidden dim
        self.K = K    # num of classes

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K))
        '''
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh())
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid())
        '''
        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L, self.K),
            nn.Sigmoid()
            #nn.Linear(self.L, 128),
            #nn.ReLU(),
            #nn.Dropout(p=0.5),
            #nn.Linear(128, self.K)
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = x
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def get_loss(self, pY_prob, Y):
        pY_prob = torch.clamp(pY_prob, min=1e-3, max=1. - 1e-3)
        p_neg_log_likelihood = -1. * (
                    Y * torch.log(pY_prob) + (1. - Y) * torch.log(1. - pY_prob))  # negative log bernoulli
        tot_loss = p_neg_log_likelihood
        return tot_loss





