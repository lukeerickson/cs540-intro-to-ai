from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # load dataset from Iris_64x64.npy
    x = np.load(filename)
    # center data around the origin
    #x = np.array([[1,2,5],[3,4,7]])
    x = x - np.mean(x, axis=0)
    # return data as a numpy array of floats
    return x

def get_covariance(dataset):
    # calculate covariance matrix of the dataset
    x = dataset
    lenX = len(x)

    x = np.transpose(x)
    x = np.dot(x, np.transpose(x))
    x = x / (lenX - 1)

    return x

def get_eig(S, m):
    # perform eigendecomposition on the covariance matrix S

    values, vectors = eigh(S, subset_by_index=[len(S)-m, len(S)-1])
    
    # formatting
    formattedValues = np.array([[values[1], 0], [0, values[0]]])
    vectors[:, [1, 0]] = vectors[:, [0, 1]]
    
    #print(type(formattedValues))
    
    return formattedValues, vectors

def get_eig_prop(S, prop):
    # the trace of a matrix is equivalent to the sum of its eigenvalues
    sum = S.trace()
    values = eigh(S, eigvals_only=True, subset_by_index=[0, len(S)-1])
    
    #x = np.empty([2, 2])
    
    arr = np.array([])
        
    startIndex = 0

    for index, item in enumerate(values):
        if((item / sum) > prop):
            arr = np.append(arr, [item])
            if(startIndex == 0):
                startIndex = index
    
    values, vectors = eigh(S, subset_by_index=[startIndex, len(S)-1])
    
    
    #print(type(vectors))
    
    values = np.reshape(values, (len(values), len(values)))
    
    #print(values)
    #print(type(values))
    
    return values, vectors
    

def project_image(image, U):
    # project each dx1 image into an m-dimensional subspace
    # this subspace is spanned by m vectors of size dx1
    # image is 4096 x 1 
    # U is 4096 x 2
    
    alpha = np.dot(np.transpose(U), image)
    recon = np.dot(U, alpha)
    
    # want dot product of U, UT, & image
    
    # return the new representation as a dx1 matrix
    return recon

def display_image(orig, proj):
    # use matplotlib to display original image and projected image side-by-side
    #print(orig)
    orig = np.reshape(orig, (64, 64))
    proj = np.reshape(proj, (64, 64))
    
    #print(orig)
    
    # Please use the format below to ensure grading consistency
    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    
    ax1.set_title('Original')
    ax2.set_title('Projection')
    
    pos1 = ax1.imshow(orig, aspect='equal')
    pos2 = ax2.imshow(proj, aspect='equal')
    
    fig.colorbar(pos1, ax = ax1, location='right')
    fig.colorbar(pos2, ax = ax2, location='right')
    
    return fig, ax1, ax2
"""
def main():
    dataset = load_and_center_dataset('Iris_64x64.npy')
    #print(len(x))
    #print(len(x[0]))
    #print(np.average(dataset))
    covMatrix = get_covariance(dataset)
    #print(len(covMatrix))
    #print(len(covMatrix[0]))
    
    diagonalMatrix, columnMatrix = get_eig(covMatrix, 2)

    #print(diagonalMatrix)
    #print(columnMatrix)
    
    diagonalMatrix, columnMatrix = get_eig_prop(covMatrix, .07)
    
    projection = project_image(dataset[50], columnMatrix)
    #print(projection)
    
    fig, ax1, ax2 = display_image(dataset[50], projection)
    plt.show()
    
main()
"""
