import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def single_element_loss(sim_matrix,N):
    
    temp = 0.4
    modified_sim_matrix = sim_matrix/temp
    modified_sim_matrix = torch.exp(modified_sim_matrix)
    #print(modified_sim_matrix)
    forward_pairs_similarities_numerator = modified_sim_matrix[torch.arange(0,N),torch.arange(N,2*N)]
    forward_pairs_similarities_deno =  torch.sum(modified_sim_matrix[torch.arange(0,N)],dim=1)
    loss_forward = -torch.log(forward_pairs_similarities_numerator/forward_pairs_similarities_deno)
    #print(f"Forward : {forward_pairs_similarities}")
    backward_pairs_similarities_numerator = modified_sim_matrix[torch.arange(N,2*N),torch.arange(0,N)] 
    backward_pairs_similarities_deno =  torch.sum(modified_sim_matrix[torch.arange(N,2*N)],dim=1)
    loss_backward = -torch.log(backward_pairs_similarities_numerator/backward_pairs_similarities_deno)
    #print(f"Backward : {backward_pairs_similarities}")

    final_loss = torch.cat((loss_forward,loss_backward)).mean()
    return final_loss


def NXT_ent_loss(z,batch_size):
    sim_matrix = torch.matmul(z , z.T)
    sim_matrix.fill_diagonal_(-torch.inf)
    return single_element_loss(sim_matrix=sim_matrix,N=batch_size)


if __name__== "__main__":
    single_element_loss(torch.randn(6,6),3)
