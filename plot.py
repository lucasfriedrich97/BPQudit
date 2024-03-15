import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib


x = np.array([2,4,6,8,10])

for model in [1,2,3,4]:

	Color = ['red','blue','green','gray','orange']

	matplotlib.rcParams.update({'font.size': 20})
	fig, axs = plt.subplots(1,2, figsize=(10, 4))

	ind = 0

	for n in [3,4]:
		indColor = 0
		for l in [10,15,20,25,30]:
			y = np.loadtxt('./data/data_model{}/var_n_{}_l_{}.txt'.format(model,n,l))
			
			if n == 3:
				axs[ind].plot(x,y,color=Color[indColor])
			else:
				axs[ind].plot(x,y,color=Color[indColor],label='L:{}'.format(l))

			axs[ind].set_xlabel("d'")
			axs[ind].set_title("{} Qudits".format(n))
			axs[ind].set_xticks([2,4,6,8,10])
			if n == 3:
				axs[ind].set_ylabel("Var[$\partial_{k}C$]")

			indColor+=1
		ind+=1


	plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)

	plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5),fontsize=20)



	ax1 = plt.axes([0.18, 0.6, 0.25, 0.25])  # Posição e tamanho do gráfico menor

	n=3
	ss=0
	x1 = np.array([4,6,8,10])
	for l in [15,20,25,30]:
		y = np.loadtxt('./data/data_model{}/var_n_{}_l_{}.txt'.format(model, n, l))
		
		ax1.plot(x1, y[1:],'-x', color=Color[1+ss],linewidth=2)
		ss+=1


	ax2 = plt.axes([0.59, 0.6, 0.25, 0.25])

	n=4
	ss=0
	x1 = np.array([4,6,8,10])
	for l in [15,20,25,30]:
		y = np.loadtxt('./data/data_model{}/var_n_{}_l_{}.txt'.format(model, n, l))
		
		ax2.plot(x1, y[1:],'-x', color=Color[1+ss],linewidth=2)
		ss+=1



	plt.show()

