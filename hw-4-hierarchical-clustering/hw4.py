from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import csv

def load_data(filepath):

    with open(filepath) as file:
        csv_reader= csv.DictReader(file)
        data = [row for row in csv_reader]
    
    return data

def calc_features(row):
    
    arr = np.arange(6, dtype = np.float64)   
    i = 0
    
    for index in row:        
        if(index != 'Country'):
            if(index != ''):
                arr[i] = row[index]
                i += 1
                
    return arr

def normalize_features(features):
    #mean = [None] * len(features)
    #std = [None] * len(features)
    
    sum = list(range(6))

    
    #print(len(features))
    
    
    for row in features:
        index = 0
        for data_pt in row:
            sum[index] += data_pt
            index += 1
    
    #mean = mean / len(features)
    mean = [x / len(features) for x in sum]
    print(mean)
    
    print("")
    
    variance = list(range(6))
    
    for row in features:
        index = 0
        for data_pt in row:
            variance[index] += ((data_pt - mean[index]) ** 2)

    variance[:] = [x / len(features) for x in variance]
    print(variance)
    std = [x ** .5 for x in variance]
    
    for row in features:
        index = 0
        for data_pt in row:
            data_pt = (data_pt - mean[index]) / std[index]
    
    # return numpy array
    return features



def hac(features):
    
    n = len(features)
    
    
    arr = np.zeros((n-1,4))
    
    # complete-linkage
    # so want farthest distance from any 2 points
    # numpy.linalg.norm() --> distance b/w pts
    # points in 6 dimensional space?
    
    # each pt starts as its own cluster
    # merge clusters as we go on
    
    # distance matrix
    distance_matrix = np.zeros((n, n))
    
    for x in range(n):
        for y in range(n):
            distance_matrix[x][y] = np.linalg.norm(features[x]-features[y])

    for z in range(n-1):
        # find closest pair of clusters
        minDist = float('inf')
        clust1 = -1
        clust2 = -1
        for x in range(n):
            for y in range(n):
                dist = distance_matrix[x][y]
                if dist < minDist and dist != 0:
                    minDist = dist
                    clust1 = x
                    clust2 = y
        if(clust1 < clust2):
            arr[z,0] = clust1
            arr[z,1] = clust2
        else:
            arr[z,0] = clust2
            arr[z,1] = clust1
        if(clust1 == clust2):
            # how to find index of 2nd cluster
            continue
    
    # you're choosing the farthest away points given the clusters you have
    
    
    # b) how to compute complete linkage distance
    # distance between elements that are farthest away from each other
    
    # how do we know how many countries are in a cluster?
    # helps me understand
    # - how to find second index of a cluster
    # - how to compute linkage distance --> find farthest away points
        
    # once we cluster 2 points, set distance between them to infinity
    
    # get pair of closest clusters and merge them
    # find smallest value in distance matrix (if not -1)
    # find smallest at first step to merge point
    # find larger at next step to figure out which
    # point in cluster to merge from
    
    # set value in distance matrix to -1
    
    # repeat
    # when do we stop
    
    

    
    
    return arr

def fig_hac(Z, names):

    # return matplotlib figure
    return
    



def main():
    data = load_data("countries.csv")
    #print(data[0])
    
    #print(calc_features(data[0]))
    #calc_features(data[0])

    
    country_names = [row["Country"] for row in data]
    features = [calc_features(row) for row in data]

    #mean = sum(features[0]) / len(features)
    #print(mean)
    
    #print(type(features))
    #print(features[0])
    
    features_normalize = normalize_features(features)
    """
    n = len(features)
    Z_raq = hac(features[:n])
    
    
    Z_normalize = hac(features_normalize[:n])
    
    fig = fig_hac(Z_raq, country_names[:n])
    plt.show()
    """
    
main()
