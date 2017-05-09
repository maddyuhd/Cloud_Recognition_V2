import numpy as np
import math
from features import sifttest as feat
import cPickle
from score import getScores

# ------------------------------------------------------------------------------------------------------------
import time as t

start =t.time()

with open('filelist.pkl', 'rb') as fid:
    fileList = cPickle.load(fid)

with open('doc.pkl', 'rb') as fid:
    doc = cPickle.load(fid)

with open('tree.pkl', 'rb') as fid:
    tree = cPickle.load(fid)

with open('nodes.pkl', 'rb') as fid:
    nodes = cPickle.load(fid)

with open('imginleaves.pkl', 'rb') as fid:
    imagesInLeaves = cPickle.load(fid)

print("Time To Load : {} seconds".format(t.time()-start))
# ------------------------------------------------------------------------------------------------------------

def weight(leafID, N):
    return math.log1p(N / 1.0 * len(imagesInLeaves[leafID]))


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


def match(filename, N,fileList,dirName):  # dirName + "/" + fname
    # q is the frequency of this image appearing in each of the leaf nodes
    q = {}
    kp, des = feat(filename, 480)
    # print des
    # mad = t.time()
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
        q[key] = q[key] * weight(key, N)
        s += q[key]
        # sum(q.values())

    for key in q:
        q[key] = q[key] / s

    # print("q Time Taken: ", str(t.time() - mad))

    return getScores(q, imagesInLeaves,doc,fileList,dirName)