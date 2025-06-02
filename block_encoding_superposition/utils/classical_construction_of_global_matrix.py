import numpy as np
from scipy.sparse import coo_matrix


def lk():
	"""
	Assemble element wise stiffness matrix
	"""
	E=1
	nu=0.3 # Poisson's ratio
	k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
	KE = E/(1-nu**2)*np.array([
	[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
	[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
	[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
	[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
	[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
	[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
	[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
	[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])


	return (KE)

def construct_global_matrices_classically(nelx,nely, rmin, penal, Emin, Emax, ndof, fixed_DOFs= None):
	KE= lk()# Element wise stiffness matrix
	# The edofMat stores the eight DOFs (two at each element's corner) 
	# for all elements (i-th element's DOFs in i-th row, start counting from the lower left corner clockwise, number of global DOFs)
	edofMat=np.zeros((nelx*nely,8),dtype=int)
	for elx in range(nelx):
		for ely in range(nely):
			el = ely+elx*nely
			n1=(nely+1)*elx+ely
			n2=(nely+1)*(elx+1)+ely
			edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
		
	# Construct the index pointers for the coo format
	# iK(j), jK(j) --> (i,j)-th entry of stiffness matrix of element e
	iK = np.kron(edofMat,np.ones((8,1))).flatten() #each row is repeated 8 times
	jK = np.kron(edofMat,np.ones((1,8))).flatten() #each number is repeated 8 times



	# Filter: Build (and assemble) the index+data vectors for the coo matrix format
	# -------------------------------------------------------------------------
	nfilter=int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2)) #2*2*3**2=36	

	iH = np.zeros(nfilter)
	jH = np.zeros(nfilter)
	sH = np.zeros(nfilter)

	cc=0
	for i in range(nelx):
			for j in range(nely):
				row=i*nely+j
			
				kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
				kk2=int(np.minimum(i+np.ceil(rmin),nelx))
				ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
				ll2=int(np.minimum(j+np.ceil(rmin),nely))
				for k in range(kk1,kk2):
					for l in range(ll1,ll2):
						col=k*nely+l
				
						#weight factor
						fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l))) #rmin - (distance between element i and j)
					
						iH[cc]=row
						jH[cc]=col
						sH[cc]=np.maximum(0.0,fac)
						cc=cc+1	

	# Finalize assembly and convert to csc format
	H=coo_matrix((sH,(iH,jH)),shape=(nelx*nely,nelx*nely)).tocsc()	

	Hs=H.sum(1)	# in the i-th entry the effect of ith element on other elements

	dofs=np.arange(2*(nelx+1)*(nely+1))
	if fixed_DOFs is None:
		fixed_DOFs=np.union1d(dofs[0:2*(nely+1):2],np.array([2*(nelx+1)*(nely+1)-1]))
	else:
		fixed_DOFs= np.array(fixed_DOFs)
	free=np.setdiff1d(dofs,fixed_DOFs)


	print('fixed DOFs: ', fixed_DOFs)
	print('free DOFs: ', free)

	all_matrices_only_free_DOFs = []
	all_matrices_only_free_DOFs_not_rescaled = []
	length= nely*nelx

	min_eigenvalues_of_all_matrices= []

	for i in range(2**length):
			config_str = format(i, f"0{length}b")
			array = np.array([int(b) for b in config_str]) 
			
			array_with_Emin= Emin+array**penal*(Emax-Emin)

			sK=((KE.flatten()[np.newaxis]).T*array_with_Emin).flatten(order='F')

			# Generate sparse global stiffness matrix in coordinate format
			# This generates K[i[k], j[k]] = sK[k]
			K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()

			
			matrix_K=K.toarray()
			np.set_printoptions(linewidth=400)

			matrix_K_not_rescaled= matrix_K/((np.max(np.abs(np.linalg.svd(KE)[1])))*2*4)

			# Optional anzeigen
			#print(f"\n[{config_str}] Global stiffness matrix as output of QSVT (not rescaled):")
			#print(np.round(matrix_K, 3))

			# Remove constrained dofs from matrix
			K_free_DOFs = matrix_K[free,:][:,free]
			
			eigenvalues= np.linalg.eigvals(K_free_DOFs)

			threshold = 1e-10
			filtered = [ev.real for ev in eigenvalues if ev.real > threshold]

			if filtered:
				min_val = min(filtered)
				#print("Minimaler Eigenwert über Threshold:", min_val)
				min_eigenvalues_of_all_matrices.append(min_val)
			
			all_matrices_only_free_DOFs.append(K_free_DOFs)
			all_matrices_only_free_DOFs_not_rescaled.append(matrix_K_not_rescaled)
	return all_matrices_only_free_DOFs, all_matrices_only_free_DOFs_not_rescaled, free

