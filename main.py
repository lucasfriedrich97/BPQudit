import model as md
import gates as ops
from tqdm import tqdm
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time 


def Var(model,modelID,name):

	if not os.path.exists('./{}/data_model{}'.format(name,modelID)):
	  	os.mkdir('./{}/data_model{}'.format(name,modelID))

	
	dim = [2,4,6,8,10]
	for n in [3,4]:
	    for l in [10,20,30,40,50]:
	        
	        y = []
	        for d in dim:
	            var = []
	            tp = trange(2000)
	            for k in tp:
	            	net = model(n,l,d)
	            	
	            	tp.set_description(f" Model:{modelID} N:{n} L:{l}, Dim:{d}")
	            		
	            	loss = net()
	            	dw=0
	            	for i in net.parameters():
	            		dw=torch.autograd.grad(loss[0][0],i)
	            		break


	            	var.append(dw[0].item())

	            y.append(np.var(var))

	        np.savetxt('./{}/data_model{}/var_n_{}_l_{}.txt'.format(name,modelID,n,l),y)


###########################

name = 'data'

if not os.path.exists('./{}'.format(name)):
	os.mkdir('./{}'.format(name))


Var(md.Model1,1,name)

Var(md.Model2,2,name)

Var(md.Model3,3,name)

Var(md.Model4,4,name)
