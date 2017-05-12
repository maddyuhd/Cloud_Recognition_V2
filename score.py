import math
from threading import Thread, current_thread
from Queue import Queue
import os


dirName = 'data/full'
fileList = sorted(os.listdir(dirName))

split = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
num_theads = min(35, len(fileList))
# print num_theads
n = 5


def lowest_val(d, v):
    result = []

    for c in range(0, v):
        m = min(d, key=d.get)
        result.append(m)
        del d[m]
    return result


def test_results(q, que, fileList_batch, imagesInLeaves, doc, scores):

    while not que.empty():

        try:
            work = que.get()
            # print current_thread().name
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

    # for idx, fname in enumerate(fileList):
    #     que.put((idx, fname))

    for i in xrange(0, len(data_split), n):

        if (len(data_split) % n == 0):
            for thr in range(i, i + n):
                # print "Normal"
                que.put(thr)
                Thread(target=test_results, args=(q, que, data_split[thr], imagesInLeaves, doc, scores)).start()
                que.join()
        else:
            if (i != int((len(data_split) - 1) / n) * n):
                for thr in range(i, i + n):
                    que.put(thr)
                    Thread(target=test_results, args=(q, que, data_split[thr], imagesInLeaves, doc, scores)).start()
                    que.join()
            else:
                for thr in range(i, len(data_split), 2):
                    que.put(thr)
                    que.put(thr + 1)
                    # que.put(thr + 2)
                    Thread(target=test_results, args=(q, que, data_split[thr], imagesInLeaves, doc, scores)).start()
                    Thread(target=test_results, args=(q, que, data_split[thr + 1], imagesInLeaves, doc, scores)).start()
                    # Thread(target=test_results, args=(q, que, data_split[thr + 2], imagesInLeaves, doc, scores)).start()
                    que.join()
        que.join()

    # # worker1.setDaemon(True)
    # # worker1.start()

    # print type(scores)
    # for eve in scores:
    #     print eve[-8:-4]

    return lowest_val(scores, 4)
