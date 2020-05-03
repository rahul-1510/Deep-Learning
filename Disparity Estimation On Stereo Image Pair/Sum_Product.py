import numpy as np
import matplotlib.pyplot as plt

## Loading Left and Right Reference Images ##
x = plt.imread('L.png')
y = plt.imread('R.png')

## Parameters ##
sig,delt,gam,k = 0.2,0.8,0.2,1

## Compatibility Functions ##
def psi(i,j,ds):
	return np.exp((-1/(2*sig*sig))*((x[i][j]-y[i+ds][j])**2))

def psi_st(ds,dt):
	return np.exp((-1/(2*gam*gam))*min(((ds-dt)**2),(delt**2)))

## Messeges ##
def mu_ts(i,j,m,n):
	tot = 0
	for dt in range(10):
		tot = tot + psi_st(y[i][j],y[m+dt][n])*psi(m,n,dt)
	return tot
def mu_s(i,j):
	prod = 1
	max = 0
	m = i
	n = j
	for ds in range(10):
		a = mu_ts(i+ds,j,i+ds-1,j)
		b = mu_ts(i+ds,j,i+ds+1,j)
		c = mu_ts(i+ds,j-1,i+ds,j)
		d = mu_ts(i+ds,j+1,i+ds,j)
		prod = a*b*c*d
		arr = psi(m,n,ds)*prod
		if(arr>max):
			max = arr
	return max

##################################################
#### We can run program for 200 * 200 pixels ####
#### Taking one Minute More (Output Attached with the assignment) ####

# f = np.zeros((200,200))
# for i in range(28,228):
# 	for j in range(28,228):
# 		f[i-28][j-28] = mu_s(i,j)
# 	print(i)

###################################################

## Storing Values ##
f = np.zeros((128,128))
for i in range(64,192):
	for j in range(64,192):
		f[i-64][j-64] = mu_s(i,j)


## Showing Image ##
plt.imshow(f,cmap='gray')
plt.show()
