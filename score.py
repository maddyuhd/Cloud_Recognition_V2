import math
from threading import Thread, current_thread
from Queue import Queue
import os


dirName = 'data/full'
fileList = sorted(os.listdir(dirName))

split = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
num_theads = min(70, len(fileList))
# print num_theads


def lowest_val(d):
    result = []

    m = min(d, key=d.get)
    result.append(m)
    del d[m]

    m = min(d, key=d.get)
    result.append(m)
    del d[m]

    m = min(d, key=d.get)
    result.append(m)
    del d[m]

    m = min(d, key=d.get)
    result.append(m)
    del d[m]

    return result


def test_results(q, que, fileList_batch, imagesInLeaves, doc, scores):  # , currimg, curr):
    # global lock  # scores
    while not que.empty():
        # lock.acquire()
        try:
            work = que.get()
            print current_thread().name
            # print "Work at batch {} ".format(work)
            for file in fileList_batch:
                img = dirName + "/" + file
                scores[img] = 0
                for leafID in imagesInLeaves:
                    if leafID in doc[img] and leafID in q:
                        scores[img] += math.fabs(q[leafID] - doc[img][leafID])
                    elif leafID in q and leafID not in doc[img]:
                        scores[img] += math.fabs(q[leafID])
                    elif leafID not in q and leafID in doc[img]:
                        scores[img] += math.fabs(doc[img][leafID])

        except Exception as e:
            raise
            # print "error d : {}".format(e)
        # finally:
            # lock.release()
        que.task_done()

    # return scores


def getScores(q, imagesInLeaves, doc, fileList, dirName):

    # global lock  # scores ,
    scores = {}
    # n = 0
    # curr = [float("inf"), float("inf"), float("inf"), float("inf")]
    # currimg = ["", "", "", ""]

    data_split = split(fileList, num_theads)
    que = Queue(maxsize=0)

    # for idx, fname in enumerate(fileList):
    # que.put((idx, fname))

    # global que
    for thrx in xrange(0, (len(data_split) - 1)):
        # for fname in data_split[thrx]:
            # que.put(fname)
        que.put(thrx)
        # for fname in data_split[thrx]:
        # que.put(fname)
        worker = Thread(target=test_results, args=(q, que, data_split[thrx], imagesInLeaves, doc, scores))  # , currimg, curr))
        # worker.setDaemon(True)
        worker.start()
        # que.join()

    que.join()
    # print type(scores)
    return lowest_val(scores)

    # for fname in fileList:
