import torch
import numpy as np
import torch.nn as nn
import copy
from collections import defaultdict

"""
Implementation by Bogdan Mazoure
Paper: https://arxiv.org/abs/2003.00722
"""

class Tau(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim,32),
            nn.Tanh(),
            nn.Linear(32,1),
            nn.Softplus()
        )
    def forward(self,x):
        return self.net(x)

def fit_tau(X_t,X_tp1, tau, tau_t, vee, opt_tau, opt_vee, device, epoch):
    """
    Meant to be called on a batch of states to estimate the batch gradient wrt tau and v.
    """
    alpha_t = 1. / np.sqrt(epoch+1)
    lam = 10

    tau_xt = tau(X_t)
    tau_xtp1 = tau(X_tp1)
    tau_t_xt = tau_t(X_t)
    tau_t_xtp1 = tau_t(X_tp1)
    v = vee

    grad_theta_tau_xt, grad_theta_tau_xtp1 = defaultdict(list),defaultdict(list)

    for i in range(len(X_t)):
        opt_tau.zero_grad()
        tau_xt.backward([torch.FloatTensor([[1] if i==j else [0] for j in range(len(tau_xt))]).to(device)],retain_graph=True)

        for param in tau.named_parameters():
            grad_theta_tau_xt[param[0]].append(param[1].grad.clone())

        opt_tau.zero_grad()
        tau_xtp1.backward([torch.FloatTensor([[1] if i==j else [0] for j in range(len(tau_xtp1))]).to(device)],retain_graph=True)
        for param in tau.named_parameters():
            grad_theta_tau_xtp1[param[0]].append(param[1].grad.clone())
        
    opt_tau.zero_grad()
    opt_vee.zero_grad()

    avg_grad_J_tau = []
    avg_grad_J_v = []

    for param in tau.named_parameters():
        """
        grad_theta: n_batch x n_out x n_in (matrix)
                    n_batch x n_out (bias)
        """
        grad_theta_tau_xt_MAT = torch.stack(grad_theta_tau_xt[param[0]])
        grad_theta_tau_xtp1_MAT = torch.stack(grad_theta_tau_xtp1[param[0]])

        """
        Defined both gradients as in Eq.17
        """

        if len(grad_theta_tau_xt_MAT.shape) == 3: # Matrix
            tiled_tau_xt =  tau_xt.repeat(grad_theta_tau_xt_MAT.shape[1],1,grad_theta_tau_xt_MAT.shape[2]).permute(1,0,2)
            tiled_tau_xtp1 =  tau_xtp1.repeat(grad_theta_tau_xtp1_MAT.shape[1],1,grad_theta_tau_xtp1_MAT.shape[2]).permute(1,0,2)
            tiled_tau_t_xt =  tau_t_xt.repeat(grad_theta_tau_xt_MAT.shape[1],1,grad_theta_tau_xt_MAT.shape[2]).permute(1,0,2)
            tiled_tau_t_xtp1 =  tau_t_xtp1.repeat(grad_theta_tau_xtp1_MAT.shape[1],1,grad_theta_tau_xtp1_MAT.shape[2]).permute(1,0,2)
        else: # Bias
            tiled_tau_xt =  tau_xt.repeat(1,grad_theta_tau_xt_MAT.shape[1])
            tiled_tau_xtp1 =  tau_xtp1.repeat(1,grad_theta_tau_xtp1_MAT.shape[1])
            tiled_tau_t_xt =  tau_t_xt.repeat(1,grad_theta_tau_xt_MAT.shape[1])
            tiled_tau_t_xtp1 =  tau_t_xtp1.repeat(1,grad_theta_tau_xtp1_MAT.shape[1])

        grad_J_tau = (tiled_tau_xt * grad_theta_tau_xt_MAT).mean(0) - (1 - alpha_t) * (tiled_tau_t_xt * grad_theta_tau_xt_MAT).mean(0) - alpha_t * (tiled_tau_t_xtp1 * grad_theta_tau_xtp1_MAT).mean(0) + 2*lam*v*grad_theta_tau_xt_MAT.mean(0)
        grad_J_v = - (2 * lam * (tau_xt.mean() - 1 - v))
        param[1].grad = grad_J_tau
        vee.grad = grad_J_v

        avg_grad_J_tau.append( grad_J_tau.mean().item() )
        avg_grad_J_v.append( grad_J_v.mean().item() )
        

    opt_tau.step()
    opt_vee.step()

    tau_t.load_state_dict(tau.state_dict())

    return np.mean(avg_grad_J_tau), np.mean(avg_grad_J_v)


if __name__ == "__main__":
    """
    Simple test with a 3x3 T. Its stationary vector is (0.25,0.5,0.25),
    so tau(0),tau(1),tau(2) / (tau(0)+tau(1)+tau(2)) should converge to this
    """
    P = np.array([[0.5,0.5,0],[0.25,0.5,0.25],[0,0.5,0.5]])
    n = len(P)
    X_t, X_tp1 = [], []

    for _ in range(1000):
        x_t = np.random.choice(range(n),size=1)[0]
        x_tp1 = np.random.choice(range(n),size=1,p=P[x_t])
        X_t.append([1 if x_t == j else 0 for j in range(n)])
        X_tp1.append([1 if x_tp1 == j else 0 for j in range(n)])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_t = torch.FloatTensor(np.array(X_t,dtype=float)).to(device)
    X_tp1 = torch.FloatTensor(np.array(X_tp1,dtype=float)).to(device)

    

    tau = Tau(n)
    tau_t = Tau(n)
    vee = torch.rand(1, requires_grad=True ,device=device)
    if device == 'cuda':
        tau = tau.cuda()
        tau_t = tau_t.cuda()

    tau_t.load_state_dict(tau.state_dict())

    opt_tau = torch.optim.Adam(tau.parameters(),lr=0.01)
    opt_vee = torch.optim.Adam([vee],lr=0.01)
    

    for epoch in range(100):

        avg_J_grad_tau, avg_J_grad_v = fit_tau(X_t,X_tp1, tau, tau_t, vee, opt_tau, opt_vee,device, epoch)
        

        p1 = tau(torch.FloatTensor([[1.,0.,0.]]).to(device)).detach().cpu().item()
        p2 = tau(torch.FloatTensor([[0.,1.,0.]]).to(device)).detach().cpu().item()
        p3 = tau(torch.FloatTensor([[0.,0.,1.]]).to(device)).detach().cpu().item()

        print('[Epoch %d] V: %f, Tau(0): %f, Tau(1):%f, Tau(2):%f'%(epoch,vee.item(),p1,p2,p3) )


    