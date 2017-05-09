import cv2


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def imresize(image, w, h, val=320):
    ar = w / float(h)
    if h > w:
        ar = w / float(h)
        newH = val
        newW = int(newH * ar)
    elif h < w:
        ar = h / float(w)
        newW = val
        newH = int(newW * ar)
    else:
        newH = val
        newW = val

    img = cv2.resize(image, (newW, newH))
    return img


def imd(img):
    (h, w) = img.shape[:2]
    return (h, w)


def process(path, val=480):
    src_img = cv2.imread(path, 0)
    h, w = imd(src_img)
    new_img = imresize(src_img, w, h, val)
    return new_img


def sifttest(imagepath, val=320):
    # src_img=cv2.imread(imagepath)
    src_img = process(imagepath, val)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)  # SIFT Feature extractor model
    kp, des = sift.detectAndCompute(src_img, None)
    # img = cv2.drawKeypoints(src_img, kp, None, color=(255, 0, 0))
    # cv2.imwrite("imagename_Sift.jpg", img)
    return kp, des


def surftest(imagepath):
    detector = cv2.xfeatures2d.SURF_create(400, 5, 5)
    kp, des = detector.detectAndCompute(cv2.cvtColor(cv2.imread(imagepath), cv2.COLOR_BGR2GRAY), None)
    return kp, des


def freaktest(imagepath):
    fast = cv2.FastFeatureDetector_create(49)
    freakExtractor = cv2.xfeatures2d.FREAK_create()
    src_img = cv2.imread(imagepath)
    kp = fast.detect(cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY), None)
    kp, des = freakExtractor.compute(src_img, kp)
    return kp, des


def brisktest(imagepath, val=320):
    detector = cv2.BRISK_create(50, 0, .5)
    freakExtractor = cv2.xfeatures2d.FREAK_create()
    src_img = process(imagepath, val)
    # src_img = cv2.imread(imagepath,0)
    kp = detector.detect(src_img, None)

    # kp,des= freakExtractor.compute(src_img,kp)
    kp, des = detector.compute(src_img, kp)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2RGB)
    # img = cv2.drawKeypoints(src_img, kp,None,color=(255,0,0))
    for k in kp:
        x, y = k.pt
        cv2.circle(src_img, (int(x), int(y)), 2, (50, 50, 50), thickness=1, lineType=8, shift=0)
        cv2.line(src_img, (int(x) - 2, int(y)), (int(x) + 2, int(y)), (0, 252, 248), 1)
        cv2.line(src_img, (int(x), int(y) + 2), (int(x), int(y) - 2), (0, 252, 248), 1)
    cv2.imwrite("imagename_Brisk.jpg", src_img)
    return kp, des


def brieftest(imagepath, val=320):
    # Detect the CenSurE key points
    star = cv2.xfeatures2d.StarDetector_create(7, 30)
    freakExtractor = cv2.xfeatures2d.FREAK_create()
    # brief = cv2.DescriptorExtractor_create("BRIEF")
    # brief = cv2.BriefDescriptorExtractor_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    src_img = cv2.imread(imagepath)
    h, w = imd(src_img)
    src_img = imresize(src_img, w, h, val)

    # imghsv = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
    # imghsv[:,:,2] = [[max(pixel - 25, 0) if pixel < 190 else min(pixel + 25, 255) for pixel in row] for row in imghsv[:,:,2]]
    # src_img=cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)

    kp = star.detect(cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY), None)
    for k in kp:
        x, y = k.pt
        cv2.circle(src_img, (int(x), int(y)), 2, (50, 50, 50), thickness=1, lineType=8, shift=0)
        cv2.line(src_img, (int(x) - 2, int(y)), (int(x) + 2, int(y)), (0, 252, 248), 1)
        cv2.line(src_img, (int(x), int(y) + 2), (int(x), int(y) - 2), (0, 252, 248), 1)  # cv2.imshow('ImageWindow',img)

    cv2.imwrite("imagename_Brief.jpg", src_img)
    kp, des = brief.compute(src_img, kp)

    return kp, des


# kp,des=brieftest("data/full/013.jpg")

# cv2.randn(src_img,(0),(99))
# fm=variance_of_laplacian(src_img)
# print fm
# print des
