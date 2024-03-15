import torch
import torch.nn as nn
import numpy as np
from math import log as log
import itertools
from torch.nn.parameter import Parameter
import math
import time


def Base(D, device='cpu'):
    '''
    This function is used to create a base tensor d
    '''
    base = torch.eye(D).unsqueeze(2).to(device)
    return base

def delta(x,y):
    if x == y:
        return 1.0
    else:
        return 0.0



def State(dits, dim, state=None, device='cpu'):
    '''
    This function is used for create a quantum state of d dimension
    Exemple.
    For create a state |00> the dimension 3 we use State('00',3) or State('0'*2,3)

    '''
    D = dim
    base = Base(D)
    s = torch.eye(1, dtype=torch.complex64)
    if state is None and dits is None:
        for dit in dits:
            s = torch.kron(s, base[0])

    elif state is None and type(dits) == str:
        for dit in dits:
            s = torch.kron(s, base[int(dit)])

    else:
        for dit in dits:
            if dit == 'h':
                s = torch.kron(s, state)
            else:
                s = torch.kron(s, base[int(dit)])

    s = s.to(device)
    return s

def Sx(j, k, base):
    '''
    This function creates the rotation matrix on the x-axis. It is used in rotating gate
    '''
    return torch.kron(base[j-1], base[k-1].T) + torch.kron(base[k-1], base[j-1].T) + 0*1j

def Sy(j, k, base):
    '''
    This function creates the rotation matrix on the y-axis. It is used in rotating gate
    '''
    return -1j*torch.kron(base[j-1], base[k-1].T) + 1j*torch.kron(base[k-1], base[j-1].T)

def Sz(j, k, base):
    '''
    This function creates the rotation matrix on the z-axis. It is used in rotating gate
    '''
    
    f = (2.0/(j*(j+1)))**0.5
    s = 0.0
    for k in range(0, j+1):
        s += ((-j)**delta(k,j))*torch.kron(base[k], base[k].T)
    return f*s + 0*1j
    

sigma = [Sx, Sy, Sz]



def Identity_sparse(D,device='cpu'):
    # Function used for create a sparse matrix representantion identity matrix of the d dimension

    indices = torch.arange(D).repeat(2, 1).to(device)
    values = torch.ones(D).to(device)
    eye_sparse = torch.sparse_coo_tensor(indices, values, (D, D)).to(device)

    return eye_sparse




def sparse(u, device='cpu'):
    '''
    Function used for create sparse matrix given a dense matrix u
    '''

    device = torch.device(device)
    u = u.to(device)
    nonzero_coords = torch.nonzero(u, as_tuple=False).t().to(device)
    values = u[tuple(nonzero_coords)].to(device)
    U_sparse = torch.sparse_coo_tensor(nonzero_coords, values, u.size(), device=device)
    return U_sparse


def nQudit(n,indice):
    esquerda = indice
    direita = n-indice-1
    return esquerda,direita

def sparse_kron(sparse1, sparse2, n1, n2,device='cpu'):
    '''
    Função otimizada para calcular o produto tensorial.
    Input:
        sparse1: matriz esparsa
        sparse2: matriz esparsa
        n1: dimensão da matriz esparsa sparse1
        n2: dimensão da matriz esparsa sparse2
    Output:
        Matriz esparsa resultante do produto tensorial
    '''

    # Coalesce as matrizes esparsas
    sparse1_coalesced = sparse1.coalesce()
    sparse2_coalesced = sparse2.coalesce()

    # Extraia valores e índices
    values1 = sparse1_coalesced.values()
    indices1 = sparse1_coalesced.indices()

    values2 = sparse2_coalesced.values()
    indices2 = sparse2_coalesced.indices()

    # Expandir os índices para realizar a multiplicação tensorial
    expanded_indices1 = indices1.unsqueeze(2)
    expanded_indices2 = indices2.unsqueeze(2).permute(0, 2, 1)

    # Calcular os produtos tensoriais
    pos = (expanded_indices1 * n2 + expanded_indices2).view(2, -1)
    val = (values1.unsqueeze(1) * values2.unsqueeze(0)).view(-1)

    # Criar a matriz esparsa resultante
    result = torch.sparse_coo_tensor(pos, val, size=(n1 * n2, n1 * n2)).to(device)

    return result




def cnot_qudits_Position(c,t,n,d,device='cpu'):
    values = torch.arange(d,dtype=torch.float).to(device)
    L = torch.stack(torch.meshgrid(*([values] * n)), dim=-1).to(device).reshape(-1, n)


    L[:,t]=(L[:,t]+L[:,c])%d

    tt = d**torch.arange(n-1, -1, -1, dtype=torch.float).to(device).reshape(n,1)

    lin = torch.matmul(L,tt).to(device)
    col = torch.arange(d**n,dtype=torch.float).to(device).reshape(d**n,1)
    return  torch.cat((lin, col), dim=1).to(device)



def CNOT_sparse(c,t,d,n,device='cpu'):
    # Cria uma matriz esparsa representando a porta CNOT
    D = d**n
    indices = cnot_qudits_Position(c,t,n,d,device=device)

    values = torch.ones(D).to(device)
    eye_sparse = torch.sparse_coo_tensor(indices.t(), values, (D, D),dtype=torch.complex64).to(device)

    return eye_sparse

    

class CNOT(nn.Module):
    #control: control qudit
    #target: target qudit
    #N: number of qudits
    def __init__(self, index = None, dim=3,N=2,device='cpu'):
        super(CNOT, self).__init__()
        D = dim
        U = CNOT_sparse(index[0],index[1],dim,N,device=device)

        self.register_buffer('U', U)

    def forward(self, x):
        return torch.matmul(self.U, x)



class Rot(nn.Module):
    #mtx_id: 0:Sx, 1:Sy, 2:Sz
    #j,k: indexes of the Gell-Mann matrices
    #index: index of the qudit to apply the gate
    def __init__(self, mtx_id=0, j=0, k=1, index=0,dim=3,N=2,device='cpu'):
        super(Rot, self).__init__()

        self.mtx_id = mtx_id
        self.j = j
        self.k = k
        self.device = device
        self.index = index
        self.dim = dim

        self.base = Base(dim,device=device)

        self.esq,self.dir = nQudit(N,index)

        self.S = sigma[self.mtx_id](self.j, self.k, self.base).to(device)

        self.I_esq = Identity_sparse(self.dim**self.esq,device=self.device)
        self.I_dir = Identity_sparse(self.dim**self.dir,device=self.device)

        
        self.angle = Parameter(torch.FloatTensor(1).uniform_(0, 2*math.pi))
       

    def forward(self, x):


        M = torch.matrix_exp(-0.5*1j*self.angle*self.S).to(self.device)

        M = sparse(M,device=self.device)


        U = sparse_kron(M,self.I_dir,self.dim,self.dim**self.dir,device=self.device)
        U = sparse_kron(self.I_esq,U,self.dim**self.esq,self.dim*(self.dim**self.dir),device=self.device)

        return U@x




def remove_elements(N, list, device='cpu'):
    tensor = torch.arange(N).to(device)

    list_tensor = torch.tensor(list, device=device)

    # Creates a Boolean mask indicating the elements to be removed
    mask = ~torch.isin(tensor, list_tensor)

    # Apply the mask to keep only elements not present in the list
    tensor_result = tensor[mask]

    return tensor_result

def NewState(state, index, dim,N,device='cpu'):
    indices_complementares = remove_elements(N,index,device=device)
    # Concatena os indices da lista_indices com os indices complementares
    index.sort()

    lista_indices = torch.tensor(index).to(device)
    nova_ordem_indices = torch.cat((lista_indices, indices_complementares)).to(device)

    # Permuta os qudits de acordo com a nova ordem de indices
    novo_estado_quantico = state.view((dim,) * N)
    novo_estado_quantico = novo_estado_quantico.permute(*nova_ordem_indices.tolist())
    novo_estado_quantico = novo_estado_quantico.reshape((dim**N,))

    return novo_estado_quantico.reshape(novo_estado_quantico.shape[0],1)


def pTraceB(state, da, db,device='cpu'):
    state_reshaped = state.view(da, db)
    state_conj = torch.conj(state_reshaped).to(device)
    rhoA = torch.matmul(state_reshaped, state_conj.transpose(0, 1)).to(device)
    del state_reshaped,state_conj
    return rhoA

def density_matrixT(state, index, dim, N,device='cpu'):
    da = int(dim**len(index))
    db = int( dim**( N-len(index) ) )
    state1 = NewState(state, index, dim, N,device=device)
    rhoA = pTraceB(state1, da, db,device=device)
    return rhoA


class density_matrix(nn.Module):

    def __init__(self, index=None, dim=3,N=2,device='cpu'):
        super(density_matrix, self).__init__()

        self.index = index
        self.dim = dim
        self.n = N
        self.device = device

        self.da = int(dim**len(index))
        self.db = int( dim**( N-len(index) ) )


    def forward(self, x):
        state1 = NewState(x, self.index, self.dim, self.n,device=self.device)
        rhoA = pTraceB(state1, self.da, self.db,device=self.device)
        del state1
        return rhoA


class prob(nn.Module):

    def __init__(self, index, dim=3,N=2,device='cpu'):
        super(prob, self).__init__()

        self.index = index
        self.dim = dim
        self.n = N
        self.device = device
        self.dm = density_matrix(index,dim=dim,N=N,device=device)


    def forward(self, state):

        probs = []
        for i in range(state.shape[1]):
            state_ = state[:,i]
            state_ = state_.view(state_.shape[0],-1)
            rhoA = self.dm(state_)
            p = abs(torch.diag(rhoA))
            p = p/torch.sum(p)
            probs.append(p)
        return torch.stack(probs)






