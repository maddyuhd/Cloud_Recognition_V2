# ------------------------------------------------------------------------------------------------------------

import cv2
import time as t
import os
import math
import operator
import re
import sys
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from features import sifttest as feat
import cPickle

# ------------------------------------------------------------------------------------------------------------

import psutil


def memorytes():
    print "CPU :", psutil.cpu_percent()
    print "Memory :", psutil.virtual_memory()


N = 500                                 # Number of samples to take as training set
rootDir = 'data/full'                   # The root directory of the dataset
nodes = {}                              # List of nodes (list of SIFT descriptors)
nodeIndex = 0                           # Index of the last node for which subtree was constructed
tree = {}                               # A dictionary in the format - node: [child1, child2, ..]
branches = 5                            # The branching factor in the vocabulary tree
leafClusterSize = 20                    # Minimum size of the leaf cluster
imagesInLeaves = {}                     # Dictionary in the format - leafID: [img1:freq, img2:freq, ..]
doc = {}                                #
bestN = 4                               #
result = np.array([0, 0, 0, 0])         #
maxDepth = 5
avgDepth = 0

# If the values are supplied as command line arguments
if len(sys.argv) == 3:
    branches = int(sys.argv[1])
    maxDepth = int(sys.argv[2])

model = MiniBatchKMeans(n_clusters=branches)  # The KMeans Clustering Model
sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)  # SIFT Feature extractor model
leafClusterSize = 2 * branches
fileList = sorted(os.listdir('data/full'))
dirName = 'data/full'
fileList1 = sorted(os.listdir('data/full1'))
dirName1 = 'data/full1'
# ------------------------------------------------------------------------------------------------------------


# Function to dump all the SIFT descriptors from training data in the feature space
def dumpFeatures(rootDir):

    features = []
    n = 0
    for fname in fileList:
        # print("Reading Image: " + dirName + "/" + fname)
        kp, des = feat(dirName + "/" + fname)
        for d in des:
            features.append(d)
        del kp, des
        n = n + 1
        if n >= N:
            break
    features = np.array(features)
    return features


# Function to construct the vocabulary tree
def constructTree(node, featuresIDs, depth):
    global nodeIndex, nodes, tree, imagesInLeaves, avgDepth
    tree[node] = []
    if len(featuresIDs) >= leafClusterSize and depth < maxDepth:
        # Here we will fetch the cluster from the indices and then use it to fit the kmeans
        # And then just after that we will delete the cluster
        model.fit([features[i] for i in featuresIDs])
        childFeatureIDs = [[] for i in range(branches)]
        for i in range(len(featuresIDs)):
            childFeatureIDs[model.labels_[i]].append(featuresIDs[i])
        for i in range(branches):
            nodeIndex = nodeIndex + 1
            nodes[nodeIndex] = model.cluster_centers_[i]
            tree[node].append(nodeIndex)
            constructTree(nodeIndex, childFeatureIDs[i], depth + 1)
    else:
        imagesInLeaves[node] = {}
        avgDepth = avgDepth + depth


# Function to lookup a SIFT descriptor in the vocabulary tree, returns a leaf cluster
def lookup(descriptor, node):
    D = float("inf")
    goto = None
    for child in tree[node]:
        # Difference between them and magnitude of the vector
        dist = np.linalg.norm([nodes[child] - descriptor])
        if D > dist:
            D = dist
            goto = child
    if tree[goto] == []:
        return goto
    return lookup(descriptor, goto)


# Constructs the inverted file frequency index
def tfidf(filename):
    global imagesInLeaves
    kp, des = feat(dirName + "/" + fname)
    for d in des:
        leafID = lookup(d, 0)
        if filename in imagesInLeaves[leafID]:
            imagesInLeaves[leafID][filename] += 1
        else:
            imagesInLeaves[leafID][filename] = 1
    del kp, des


# This function returns the weight of a leaf node
def weight(leafID):
    return math.log1p(N / 1.0 * len(imagesInLeaves[leafID]))


# Returns the scores of the images in the dataset
def getScores(q):
    scores = {}
    n = 0
    curr = [float("inf"), float("inf"), float("inf"), float("inf")]
    currimg = ["", "", "", ""]
    for fname in fileList:
        img = dirName + "/" + fname
        scores[img] = 0
        for leafID in imagesInLeaves:
            if leafID in doc[img] and leafID in q:
                scores[img] += math.fabs(q[leafID] - doc[img][leafID])
            elif leafID in q and leafID not in doc[img]:
                scores[img] += math.fabs(q[leafID])
            elif leafID not in q and leafID in doc[img]:
                scores[img] += math.fabs(doc[img][leafID])
            if scores[img] > curr[-1]:
                break
        if scores[img] <= curr[0]:
            currimg[3], curr[3] = currimg[2], curr[2]
            currimg[2], curr[2] = currimg[1], curr[1]
            currimg[1], curr[1] = currimg[0], curr[0]
            currimg[0], curr[0] = img, scores[img]
        elif scores[img] > curr[0] and scores[img] <= curr[1]:
            currimg[3], curr[3] = currimg[2], curr[2]
            currimg[2], curr[2] = currimg[1], curr[1]
            currimg[1], curr[1] = img, scores[img]
        elif scores[img] > curr[1] and scores[img] <= curr[2]:
            currimg[3], curr[3] = currimg[2], curr[2]
            currimg[2], curr[2] = img, scores[img]
        elif scores[img] > curr[2] and scores[img] <= curr[3]:
            currimg[3], curr[3] = img, scores[img]
        n = n + 1
        if n >= N:
            break
    return currimg


# Return the bestN best matches
def findBest(scores, bestN):
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1))
    return sorted_scores[:bestN]


def accuracy(F, M1, M2, M3, M4):
    a = [0, 0, 0, 0]
    group = int(F / 4)
    if int(M1 / 4) == group:
        a[0] = 1
    if int(M2 / 4) == group:
        a[1] = 1
    if int(M3 / 4) == group:
        a[2] = 1
    if int(M4 / 4) == group:
        a[3] = 1
    return np.array(a)

# Finds 4 best matches for the query


def match(filename):  # dirName + "/" + fname
    # q is the frequency of this image appearing in each of the leaf nodes
    q = {}
    kp, des = feat(filename, 480)
    if des is not None:
        for d in des:
            leafID = lookup(d, 0)
            if leafID in q:
                q[leafID] += 1
            else:
                q[leafID] = 1
    else:
        print "error at {}".format(filename)
    s = 0.0
    for key in q:
        q[key] = q[key] * weight(key)
        s += q[key]
    for key in q:
        q[key] = q[key] / s

    return getScores(q)
    # return findBest(scores, bestN)


def getImgID(s):
    return int((re.findall("\d+", s))[0])

# ------------------------------------------------------------------------------------------------------------


start = t.time()
print("Extracting Features: " + rootDir + " ...")
# dump all features as array
features = dumpFeatures(rootDir)
end = t.time()
print("Time Taken: ", str(round((end - start) / 60, 2)))

start = t.time()
print("Constructing Vocabulary Tree ... ")
# average of all values in row
root = features.mean(axis=0)
nodes[0] = root
# Array of indices into the construct tree function
featuresIDs = [x for x in range(len(features))]
constructTree(0, featuresIDs, 0)
end = t.time()
print("Time Taken: ", str(round((end - start) / 60, 2)))

# memorytes()

del features

avgDepth = int(avgDepth / len(imagesInLeaves))

start = t.time()
print("Mapping images to leaf nodes of the tree ...")

n = 0
for fname in fileList:
    filename = dirName + "/" + fname
    tfidf(filename)
    n = n + 1
    if n >= N:
        break

# Creating weights for the leaf images tress
for leafID in imagesInLeaves:
    for img in imagesInLeaves[leafID]:
        if img not in doc:
            doc[img] = {}
        # weight of leafId * frequency of occurance
        doc[img][leafID] = weight(leafID) * (imagesInLeaves[leafID][img])

# scale the weights in range(0,1)
for img in doc:
    s = 0.0
    for leafID in doc[img]:
        s += doc[img][leafID]
    for leafID in doc[img]:
        doc[img][leafID] /= s

end = t.time()

print("Time Taken: ", str(round((end - start) / 60, 2)))

# memorytes()

# saving stuff
# ------------------------------------------------------------------------------------------------------------

with open('model.pkl', 'wb') as fid:
    cPickle.dump(model, fid)

with open('imginleaves.pkl', 'wb') as fid:
    cPickle.dump(imagesInLeaves, fid)

with open('tree.pkl', 'wb') as fid:
    cPickle.dump(tree, fid)

with open('filelist.pkl', 'wb') as fid:
    cPickle.dump(fileList, fid)

with open('doc.pkl', 'wb') as fid:
    cPickle.dump(doc, fid)

with open('nodes.pkl', 'wb') as fid:
    cPickle.dump(nodes, fid)

# ------------------------------------------------------------------------------------------------------------


# print("Finding Best Matches for each image ...")
# start = t.time()
# n = 0
# co = 0

# for fname in fileList1:
#     filename = dirName1 + "/" + fname
#     group = match(filename)
#     print(filename, ": ", getImgID(group[0]), getImgID(group[1]), getImgID(group[2]), getImgID(group[3]))
#     # print(filename, ": ", accuracy(getImgID(filename), getImgID(group[0]), getImgID(group[1]), getImgID(group[2]), getImgID(group[3])))

# # ------------------------------------------------------------------------------------------------------------
#     name = int(filename[-7:-4])

#     if (name == int(getImgID(group[0])) or name == int(getImgID(group[1])) or name == int(getImgID(group[2])) or name == int(getImgID(group[3]))):
#         co += 1

# # ------------------------------------------------------------------------------------------------------------

#     result = result + accuracy(getImgID(filename), getImgID(group[0]), getImgID(group[1]), getImgID(group[2]), getImgID(group[3]))

#     if (100 * n / N) % 25 == 0 and (100 * n / N) != 0:
#         print(100 * n / N, "%, done ... ")
#     n = n + 1
#     if n >= N:
#         break

# end = t.time()
# # memorytes()

# print "accuracy = {}%".format((co / float(len(fileList1))) * 100)

# print "average time = {}".format((end - start) / len(fileList1))
# print("Time Taken: ", str(round((end - start) / 60, 2)))
# print(branches, maxDepth, result / N, ((result / N).sum()) / 0.04)

# # ------------------------------------------------------------------------------------------------------------
