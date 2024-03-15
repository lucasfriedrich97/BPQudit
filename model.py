import gates as ops
import torch
import torch.nn as nn
import numpy as np

class Model1(nn.Module):
    def __init__(self,n,l,dim,device='cpu'):
        super(Model1, self).__init__()
        self.state = ops.State('0'*n,dim = dim,device=device)
        self.n = n
        self.l = l
        self.dim = dim
        self.device = device
        self.layer = []
        
        for i in range(l):
            for j in range(n):
                
                mdx = np.random.choice(3)
                if mdx == 2:
                    d1 = np.random.choice(dim-1)+1
                    self.layer.append( ops.Rot(mtx_id=mdx,index=j,j=d1,k=0,N=n,dim=dim,device=device) )
                else:
                    d2 = np.random.choice(dim-1)+2
                    d1 = np.random.choice(d2-1)+1
                    self.layer.append( ops.Rot(mtx_id=mdx,index=j,j=d1,k=d2,N=n,dim=dim,device=device) )
            
            for j in range(n-1):
                self.layer.append( ops.CNOT( index=[j,j+1], dim=dim,N=n,device=device ) )
            

        self.layer.append(ops.prob(list(range(self.n)),dim=dim,N=n,device=device))
        self.modelo = nn.Sequential(*self.layer).to(device)

    def forward(self):
        x = self.state
        x = self.modelo(x)
        return x



#########################################################################################

class Model2(nn.Module):
    def __init__(self,n,l,dim,device='cpu'):
        super(Model2, self).__init__()
        self.state = ops.State('0'*n,dim = dim,device=device)
        self.n = n
        self.l = l
        self.dim = dim
        self.device = device
        self.layer = []
        
        for i in range(l):
            for j in range(n):
                
                mdx = np.random.choice(3)
                if mdx == 2:
                    d1 = np.random.choice(dim-1)+1
                    self.layer.append( ops.Rot(mtx_id=mdx,index=j,j=d1,k=0,N=n,dim=dim,device=device) )
                else:
                    d2 = np.random.choice(dim-1)+2
                    d1 = np.random.choice(d2-1)+1
                    self.layer.append( ops.Rot(mtx_id=mdx,index=j,j=d1,k=d2,N=n,dim=dim,device=device) )


                mdx = np.random.choice(3)
                if mdx == 2:
                    d1 = np.random.choice(dim-1)+1
                    self.layer.append( ops.Rot(mtx_id=mdx,index=j,j=d1,k=0,N=n,dim=dim,device=device) )
                else:
                    d2 = np.random.choice(dim-1)+2
                    d1 = np.random.choice(d2-1)+1
                    self.layer.append( ops.Rot(mtx_id=mdx,index=j,j=d1,k=d2,N=n,dim=dim,device=device) )


            for j in range(n-1):
                self.layer.append( ops.CNOT( index=[j,j+1], dim=dim,N=n,device=device ) )

        self.layer.append(ops.prob(list(range(self.n)),dim=dim,N=n,device=device))
        self.modelo = nn.Sequential(*self.layer).to(device)

    def forward(self):
        x = self.state
        x = self.modelo(x)
        return x



#################################################################################################

class Model3(nn.Module):
    def __init__(self,n,l,dim,device='cpu'):
        super(Model3, self).__init__()
        self.state = ops.State('0'*n,dim = dim,device=device)
        self.n = n
        self.l = l
        self.dim = dim
        self.device = device
        self.layer = []
        
        for i in range(l):
            for j in range(n):
                mdx = np.random.choice(3)
                if mdx == 2:
                    d1 = np.random.choice(dim-1)+1
                    self.layer.append( ops.Rot(mtx_id=mdx,index=j,j=d1,k=0,N=n,dim=dim,device=device) )
                else:
                    d2 = np.random.choice(dim-1)+2
                    d1 = np.random.choice(d2-1)+1
                    self.layer.append( ops.Rot(mtx_id=mdx,index=j,j=d1,k=d2,N=n,dim=dim,device=device) )
            
            for j in range(n):
                for k in range(n):
                    if j !=k:
                        self.layer.append( ops.CNOT( index=[j,k], dim=dim,N=n,device=device ) )

        self.layer.append(ops.prob(list(range(self.n)),dim=dim,N=n,device=device))
        self.modelo = nn.Sequential(*self.layer).to(device)

    def forward(self):
        x = self.state
        x = self.modelo(x)
        return x



#########################################################################################

class Model4(nn.Module):
    def __init__(self,n,l,dim,device='cpu'):
        super(Model4, self).__init__()
        self.state = ops.State('0'*n,dim = dim,device=device)
        self.n = n
        self.l = l
        self.dim = dim
        self.device = device
        self.layer = []
        
        for i in range(l):
            for j in range(n):
                mdx = np.random.choice(3)
                if mdx == 2:
                    d1 = np.random.choice(dim-1)+1
                    self.layer.append( ops.Rot(mtx_id=mdx,index=j,j=d1,k=0,N=n,dim=dim,device=device) )
                else:
                    d2 = np.random.choice(dim-1)+2
                    d1 = np.random.choice(d2-1)+1
                    self.layer.append( ops.Rot(mtx_id=mdx,index=j,j=d1,k=d2,N=n,dim=dim,device=device) )


                mdx = np.random.choice(3)
                if mdx == 2:
                    d1 = np.random.choice(dim-1)+1
                    self.layer.append( ops.Rot(mtx_id=mdx,index=j,j=d1,k=0,N=n,dim=dim,device=device) )
                else:
                    d2 = np.random.choice(dim-1)+2
                    d1 = np.random.choice(d2-1)+1
                    self.layer.append( ops.Rot(mtx_id=mdx,index=j,j=d1,k=d2,N=n,dim=dim,device=device) )



            for j in range(n):
                for k in range(n):
                    if j != k:
                        self.layer.append( ops.CNOT( index=[j,k], dim=dim,N=n,device=device ) )

        self.layer.append(ops.prob(list(range(self.n)),dim=dim,N=n,device=device))
        self.modelo = nn.Sequential(*self.layer).to(device)

    def forward(self):
        x = self.state
        x = self.modelo(x)
        return x



