import math
from multiprocessing import Pool

def getScores(q, imagesInLeaves,doc,fileList,dirName):

    scores = {}
    # n = 0
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
        # n = n + 1
        # if n >= N:
        #     break
    # print("score Time Taken: ", str(t.time() - mad))
    return currimg
