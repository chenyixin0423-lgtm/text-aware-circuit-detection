import cv2
import numpy as np


def has_intersection_bitwise(contour1, contour2, shape):
    mask1 = np.zeros(shape, dtype=np.uint8)
    mask2 = np.zeros(shape, dtype=np.uint8)

    cv2.drawContours(mask1, [contour1], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(mask2, [contour2], -1, 255, thickness=cv2.FILLED)

    intersection = cv2.bitwise_and(mask1, mask2)

    return np.any(intersection > 0)


def contours(str, contour, allotherContours):
    import cv2

    image = cv2.imread(str, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.bitwise_not(image)

    contours, hierarchy = cv2.findContours(
        image2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # cv2.drawContours(image, contours, -1, (0, 255, 0), -1)
    # print(contours)
    # print(hierarchy)

    h, w = image2.shape
    myshape = (h, w)
    mask1 = np.zeros(myshape, dtype=np.uint8)
    mask1 = cv2.bitwise_not(mask1)
    usefull_contours = []
    for i in range(len(contours)):
        x = has_intersection_bitwise(contour, contours[i], myshape)

        if x == True:
            usefull_contours.append(contours[i])
    connected = []
    for i in range(len(allotherContours)):
        for j in range(len(usefull_contours)):
            x = has_intersection_bitwise(
                allotherContours[i], usefull_contours[j], myshape
            )
            if x == True:
                print(i)
                connected.append(i)
                break
    return connected
    # return usefull_contours
    # if x==True:
    #     cv2.drawContours(mask1, contours, i, (0, 255, 0), -1)
    # cv2.imshow('Contours', mask1)
    # cv2.waitKey(0)
    # intersecting_contours = []

    # for i, cnt in enumerate(contours):

    #     mask_cnt = np.zeros((h, w), dtype=np.uint8)
    #     cv2.fillPoly(mask_cnt, [cnt], 255)

    #     intersection_mask = cv2.bitwise_and(mask_ref, mask_cnt)

    #     contours, _ = cv2.findContours(intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     if contours:
    #         intersecting_contours.append(cnt)

    # image = np.zeros((500, 500, 3), dtype=np.uint8)

    # # contour_ref = contour

    # # for cnt in contours:
    # #     cv2.fillPoly(mask_all, [cnt], 255)

    # # cv2.fillPoly(mask_ref, [contour_ref], 255)

    # # intersection_mask = cv2.bitwise_and(mask_all, mask_ref)

    # # intersect_contours, _ = cv2.findContours(intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # imageN = np.zeros((h,w,3), dtype=np.uint8)

    # # cv2.drawContours(imageN, contours, -1, (255, 0, 0), 2)
    # # cv2.drawContours(imageN, [contour_ref], -1, (0, 255, 0), 2)

    # # cv2.drawContours(imageN, intersect_contours, -1, (0, 0, 255), -1)

    # # cv2.imshow("Intersection", imageN)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow('Contours', image)
    # cv2.waitKey(0)


# import numpy as np

# x1, y1, x2, y2 = [218.40405, 110.51253, 265.8917, 179.78528]
# contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
# contours("1110_0_p.png", contour)
