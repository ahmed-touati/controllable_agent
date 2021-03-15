import torch
import torch.nn.functional as F


def value_iteration(R, P, gamma, atol=0.0001, max_iteration=1000):
    q = torch.zeros_like(R)
    for i in range(max_iteration):
        q_old = q
        v = torch.max(q, dim=1)[0]
        q = R + gamma * torch.einsum('ijk, k->ij', P, v)
        if torch.allclose(q, q_old, atol=atol):
            break
    return q


def extract_policy(q, policy_type='boltzmann', temp=1, eps=0):
    action_space = q.shape[-1]
    if policy_type == 'boltzmann':
        policy = F.softmax(q / temp, dim=-1)
    elif policy_type == 'greedy':
        max_idx = torch.argmax(q, 1, keepdim=True)
        policy = torch.zeros_like(q).fill_(eps / action_space)
        policy.scatter_(1, max_idx, 1 - eps + eps / action_space)
    else:
        raise NotImplementedError()
    return policy


def compute_successor_reps(P, pi, gamma):
    state_space, action_space = P.shape[:2]
    P_pi = torch.einsum('sax, xu -> saxu', P, pi)  # S x A x S x A
    P_pi = P_pi.transpose(0, 1).transpose(2, 3).reshape(state_space * action_space,
                                                        state_space * action_space)
    Id = torch.eye(*P_pi.size(), out=torch.empty_like(P_pi))
    sr_pi = torch.inverse(Id - gamma * P_pi)
    return sr_pi