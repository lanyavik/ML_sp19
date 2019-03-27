"""
ML_hw3_problem4
Xinyue Chen (xc1305)

"""
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

######### Load the data ##########

infile = open('faces.csv','r')
img_data = infile.read().strip().split('\n')
img = [map(int,a.strip().split(',')) for a in img_data]
pixels = []
for p in img:
    pixels += p
faces = np.reshape(pixels,(400,4096))

######### Global Variable ##########

image_count = 0

######### Function that normalizes a vector x (i.e. |x|=1 ) #########

# > numpy.linalg.norm(x, ord=None, axis=None, keepdims=False) 
#   This function is able to return one of eight different matrix norms, 
#   or one of an infinite number of vector norms (described below), 
#   depending on the value of the ord parameter.
#   Note: in the given functionm, U should be a vector, not a array. 
#         You can write your own normalize function for normalizing 
#         the colomns of an array.

def normalize(U):
	return U / LA.norm(U) 


######### Display first face #########

# Useful functions:
# > numpy.reshape(a, newshape, order='C')
#   Gives a new shape to an array without changing its data.
# > matplotlib.pyplot.figure()
# 	Creates a new figure.
# > matplotlib.pyplot.title()
#	Set a title of the current axes.
# > matplotlib.pyplot.imshow()
#	Display an image on the axes.
#	Note: You need a matplotlib.pyplot.show() at the end to display all the figures.

first_face = np.reshape(faces[0],(64,64),order='F')
image_count+=1
plt.figure(image_count)
plt.title('First_face')
plt.imshow(first_face,cmap=plt.cm.gray)


########## display a random face ###########

# Useful functions:
# > numpy.random.choice(a, size=None, replace=True, p=None)
#   Generates a random sample from a given 1-D array
# > numpy.ndarray.shape()
#   Tuple of array dimensions.
#   Note: There are two ways to order the elements in an array: 
#         column-major order and row-major order. In np.reshape(), 
#         you can switch the order by order='C' for row-major(default), 
#         or by order='F' for column-major. 


#### Your Code Here ####
i=np.random.choice(range(len(faces)))
random_face = np.reshape(faces[i],(64,64),order='F')
image_count+=1
plt.figure(image_count)
plt.title('random_face(No.%s)'%(i+1))
plt.imshow(random_face,cmap=plt.cm.gray)


########## compute and display the mean face ###########

# Useful functions:
# > numpy.mean(a, axis='None', ...)
#   Compute the arithmetic mean along the specified axis.
#   Returns the average of the array elements. The average is taken over 
#   the flattened array by default, otherwise over the specified axis. 
#   float64 intermediate and return values are used for integer inputs.

#### Your Code Here ####
#||calculate mean
mean_x=np.mean(faces,axis=0)
print('mean of faces data:',mean_x)
#||display
mean_face = np.reshape(mean_x,(64,64),order='F')
image_count+=1
plt.figure(image_count)
plt.title('Mean_face')
plt.imshow(mean_face,cmap=plt.cm.gray)


######### substract the mean from the face images and get the centralized data matrix A ###########

# Useful functions:
# > numpy.repeat(a, repeats, axis=None)
#   Repeat elements of an array.

#### Your Code Here ####
A=faces-np.reshape(np.repeat(mean_x,400),(400,4096),order='F')


######### calculate the eigenvalues and eigenvectors of the covariance matrix #####################

# Useful functions:
# > numpy.matrix()
#   Returns a matrix from an array-like object, or from a string of data. 
#   A matrix is a specialized 2-D array that retains its 2-D nature through operations. 
#   It has certain special operators, such as * (matrix multiplication) and ** (matrix power).

# > numpy.matrix.transpose(*axes)
#   Returns a view of the array with axes transposed.

# > numpy.linalg.eig(a)[source]
#   Compute the eigenvalues and right eigenvectors of a square array.
#   The eigenvalues, each repeated according to its multiplicity. 
#   The eigenvalues are not necessarily ordered. 

#### Your Code Here ####
V=np.divide(np.matrix.transpose(A)*np.matrix(A),400) #||covariance matrix
#||compute its orthonormal eigenvectors by a smaller matrix L = AA'
def normcols(Umatrix):
    for U in np.array(np.matrix.transpose(Umatrix)):
        N=normalize(U)
        try:
            Narray=np.append(Narray,[N],axis=0)
        except NameError:
            Narray=np.array([N])
    return np.matrix(np.matrix.transpose(Narray))
L=np.matrix(A)*np.matrix.transpose(A)
Evalues, Evectors = np.linalg.eig(L) #||compute v by L
Evectors=np.matrix.transpose(A)*np.matrix(Evectors) #||z=A'*v
Evectors=normcols(Evectors) #||normalize z
#|| a very small positive eigenvalue here suffers float-point error
#|| and turn negative
'''check computation
V=np.matrix.transpose(A)*np.matrix(A)
print(Evalues[i])
print(type(Evectors))
print((V * Evectors[:,i])/Evectors[:,i])
'''

########## Display the first 16 principal components ##################

#### Your Code Here ####
indices=np.argsort(Evalues)
EVs=Evalues[indices[::-1]] 
PCs=np.matrix.transpose(Evectors)[indices[::-1]]
print('first 16 principle components data:\n',PCs[:16])
for idx in range(16):
    image_count+=1
    plt.figure(image_count)
    plt.title('PC %s'%(idx+1))
    plt.imshow(np.reshape(PCs[idx],(64,64),order='F'),cmap=plt.cm.gray)


########## Reconstruct the first face using the first two PCs #########

#### Your Code Here ####
def reconstruction(No,k):
    U_t=PCs[:k]
    W=np.matrix(U_t)*np.matrix.transpose(np.matrix(faces[No-1]\
               -mean_x))
    aprox_face=np.matrix.transpose(np.matrix(mean_x))\
               +np.matrix.transpose(U_t)*W
    return np.reshape(aprox_face,(64,64),order='F')

#||display the reconstructed first face
image_count+=1
plt.figure(image_count)
plt.title('reconstructed_first_face by 2 PCs')
plt.imshow(reconstruction(1,k=2),cmap=plt.cm.gray)

########## Reconstruct random face using the first 5, 10, 25, 50, 100, 200, 300, 399  PCs ###########

#### Your Code Here ####
i=99 #||reconstruct image No.100 (index 99)
image_count+=1
fig1=plt.figure(image_count)
fig1.suptitle('reconstructed_No.%s_face'%(i+1))
sub_count_2=0
for K in [5, 10, 25, 50, 100, 200, 300, 399]:
    sub_count_2+=1
    plt.subplot(240+sub_count_2)
    plt.title('%s PCs'%K)
    plt.imshow(reconstruction(i+1,K),cmap=plt.cm.gray)
    '''check
    image_count+=1
    plt.figure(image_count)
    plt.title('original_No.%s_face'%(i+1))
    plt.imshow(np.reshape(faces[i],(64,64),order='F')\
               ,cmap=plt.cm.gray)
    '''
fig1.savefig('P4(7)_plot8faces.pdf')


######### Plot proportion of variance of all the PCs ###############

# Useful functions:
# > matplotlib.pyplot.plot(*args, **kwargs)
#   Plot lines and/or markers to the Axes. 
# > matplotlib.pyplot.show(*args, **kw)
#   Display a figure. 
#   When running in ipython with its pylab mode, 
#   display all figures and return to the ipython prompt.

#### Your Code Here ####
image_count+=1
fig2=plt.figure(image_count) 
PCprops=EVs/np.sum(EVs)
print('propertion of variance data:\n',PCprops)
#||note the last propertion is negative due to floatpoint error
plt.plot(PCprops)
plt.xlabel('Nth principle component')
plt.ylabel('proportion of variance(%)')
fig2.savefig('P4(8)_plotProportion.pdf')


