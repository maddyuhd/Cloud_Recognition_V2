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

    for c in range(0, 4):
        m = min(d, key=d.get)
        result.append(m)
        del d[m]
    return result


def test_results(q, que, fileList_batch, imagesInLeaves, doc, scores):

    while not que.empty():

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
            # raise
            print "error d : {}".format(e)

        finally:
            que.task_done()


def getScores(q, imagesInLeaves, doc, fileList, dirName):

    scores = {}

    data_split = split(fileList, num_theads)
    que = Queue(maxsize=0)

    # print data_split[12 - 1]

    # for idx, fname in enumerate(fileList):
    #     que.put((idx, fname))
    n = 5

    for i in xrange(0, len(data_split), n):

        if (len(data_split) % n == 0):
            # print "normal"
            for thr in range(i, i + n):
                que.put(thr)
                print thr
                Thread(target=test_results, args=(q, que, data_split[thr], imagesInLeaves, doc, scores)).start()
                que.join()
        else:
            if (i != int((len(data_split) - 1) / n) * n):
                for thr in range(i, i + n):
                    que.put(thr)
                    Thread(target=test_results, args=(q, que, data_split[thr], imagesInLeaves, doc, scores)).start()
                    que.join()
            else:
                for thr in range(i, len(data_split)):
                    que.put(thr)
                    Thread(target=test_results, args=(q, que, data_split[thr], imagesInLeaves, doc, scores)).start()
                    que.join()

        # que.put(thr)

    # # worker1.setDaemon(True)
    # # worker1.start()

    # print type(scores)
    # for eve in scores:
    #     print eve[-8:-4]

    # print "value of 227",scores["data/full/0227.jpg"]
    # print "value of 272",scores["data/full/0271.jpg"]
    return lowest_val(scores)
