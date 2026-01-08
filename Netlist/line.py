def getLines(str):

    import cv2
    import numpy as np

    # Load the CAD image
    image = cv2.imread(str, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 100)
    cv2.imwrite(str+".png", edges)

    # revert black and white
    # edges = cv2.bitwise_not(image)
    # cv2.imwrite(str+".png", edges)
    # cv2.imshow('Line Detection', edges)
    # cv2.waitKey(0)
    # Hough Line Transform
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=40, minLineLength=10, maxLineGap=15
    )
    # print(lines)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            cv2.line(image, (x1, y1), (x2, y2), 50, 5)
    #         # cv2.circle(image, (x1, y1), 10, (0, 0, 0), -1)
    #         # cv2.circle(image, (x2, y2), 10, (0, 0, 0), -1)
    cv2.imwrite(str+".jpg", image)
    # cv2.imshow('Line Detection', image)
    # cv2.waitKey(0)
    return lines

# import hough.htlcnn
# from hough.htlcnn.demo import predict_lines
# def newGetLines(str):
#     import cv2
#     image = cv2.imread(str)
#     line_marked_img, processed_lines = predict_lines(config_file="./hough/htlcnn/config/wireframe.yaml", checkpoint_path="./hough/checkpoint.pth", image=image, devices="0", threshold=0.99)
#     lines = []
#     for i in range(len(processed_lines)):
#         lines.append([(int( processed_lines[i][0][1]),int(processed_lines[i][0][0]) , int(processed_lines[i][1][1]), int(processed_lines[i][1][0]),)])
#
    # if lines is not None:
    #     for line in lines:
    #         print(line)
    #         x1, y1, x2, y2 = line[0]
    #         x1 = int(x1)
    #         y1 = int(y1)
    #         x2 = int(x2)
    #         y2 = int(y2)
    #         print(x1, y1, x2, y2)
    #         cv2.line(image, (x1, y1), (x2, y2), 50, 5)
    # #         # cv2.circle(image, (x1, y1), 10, (0, 0, 0), -1)
    # #         # cv2.circle(image, (x2, y2), 10, (0, 0, 0), -1)
    # cv2.imwrite(str+"11.jpg", image)
    return lines
# # Display the result
# cv2.imshow('Line Detection', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# getLines("1001_0.png")
