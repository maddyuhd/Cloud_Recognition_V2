# ------------------------------------------------------------------------------------------------------------

import time as t
import os
import re

from match import match

# ------------------------------------------------------------------------------------------------------------
N = 500                                 # Number of samples to take as training set
rootDir = '...data/full'                   # The root directory of the dataset
nodeIndex = 0                           # Index of the last node for which subtree was constructed
leafClusterSize = 20                    # Minimum size of the leaf cluster
bestN = 4                               #
final_count = 0
# result = np.array([0, 0, 0, 0])         #

dirName = 'data/full'
fileList = sorted(os.listdir(dirName))
dirName1 = 'data/full1'
fileList1 = sorted(os.listdir(dirName1))

# ------------------------------------------------------------------------------------------------------------

def getImgID(s):
    return int((re.findall("\d+", s))[0])

def singleInput_serial(input):
    filename = dirName1 + "/" + input +".jpg"
    group = match(filename, N,fileList,dirName)
    print(filename[-9:], ": ", getImgID(group[0]), getImgID(group[1]), getImgID(group[2]), getImgID(group[3]))
    
    # Accuracy cal
    name = int(filename[-7:-4])
    final_count = 0
    if (name == int(getImgID(group[0])) or name == int(getImgID(group[1])) or name == int(getImgID(group[2])) or name == int(getImgID(group[3]))):
        final_count += 1
    print "accuracy = {}%".format((final_count ) * 100)

# ------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # singleInput_serial("0209")

    for fname in fileList1:
        filename = dirName1 + "/" + fname
        group = match(filename, N, dirName)
        print(filename[-9:], ": ", getImgID(group[0]), getImgID(group[1]), getImgID(group[2]), getImgID(group[3]))
# ------------------------------------------------------------------------------------------------------------
        name = int(filename[-7:-4])

        if (name == int(getImgID(group[0])) or name == int(getImgID(group[1])) or name == int(getImgID(group[2])) or name == int(getImgID(group[3]))):
            final_count += 1
    print "accuracy = {}%".format((final_count / float(len(fileList1))) * 100)

# ------------------------------------------------------------------------------------------------------------