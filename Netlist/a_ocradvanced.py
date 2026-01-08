import cv2
import json
from PIL import Image
import os

from ultralytics import YOLO

model = YOLO(
    "YOUR MODEL PATH"  # path to your trained model weights
)
import os
from PIL import Image
import pytesseract
from PIL import ImageDraw, ImageFont
import numpy as np
import easyocr
from a_newOCR import TextRemover


def draw_bbox_to_image(image1, str):
    reader = easyocr.Reader(["en"])
    draw = ImageDraw.Draw(image1)

    image_path = str
    image = cv2.imread(image_path)

    results = reader.readtext(image, detail=1)

    for result in results:
        bbox, text, confidence = result

        top_left = tuple(map(int, bbox[0]))

        bottom_right = tuple(map(int, bbox[2]))
        x1 = top_left[0]
        y1 = top_left[1]
        x2 = bottom_right[0]
        y2 = bottom_right[1]
        draw.rectangle([x1, y1, x2, y2], fill="white")

    return image1


correct = 0
total = 0
# files = os.listdir("../testOurs")
files =["Book5.png"]
# for file in files:
# files = ("image_9.jpg",)
for rawfile in files:

    # if rawfile[-3:] != "jpg":
    #     continue
    file = rawfile
    print(file)
    try:
        im1 = Image.open(file)
    except Exception as e:
        print(e)
        continue
    remover = TextRemover()
    im1=remover.remove_text(file)
    results = model.predict(
        source=im1, save=True, half=True, augment=True, iou=0.5
    )  # save plotted images
    result = results[0].to("cpu").numpy()

    length = len(result.boxes.xyxy)
    # map_id_to_text = {}
    # for i in range(length):
    #     print(result.boxes.xyxy[i])

    #     x1, y1, x2, y2 = result.boxes.xyxy[i]
    #     # extend the box
    #     x1 = x1 - (x2 - x1) * 0.05
    #     x2 = x2 + (x2 - x1) * 0.05
    #     y1 = y1 - (y2 - y1) * 0.2
    #     y2 = y2 + (y2 - y1) * 0.2
    #     import crop_OCR

    #     tempA = crop_OCR.crop(file, [x1, y1, x2, y2])
    #     # if tempA == None:
    #     #     continue
    #     map_id_to_text[i] = tempA

    # print(result.boxes)
    con = {}
    for i in range(length):
        con[i] = result.boxes.xyxy[i]
    print(con)
    # NEW ONE
    names = {
        0: "pmos",
        1: "nmos",
        2: "NPN",
        3: "PNP",
        4: "indu",
        5: "Diode",
        6: "R",
        7: "C",
        8: "ground",
        9: "V",
        10: "I",
    }

    map_id_to_name = []

    for i in range(length):
        map_id_to_name.append(names[int(result.boxes.cls[i])])

    imopen = Image.open(file)
    testdraw = ImageDraw.Draw(imopen)
    con_set = set()
    for i in range(length):
        # print("newone")
        # print(names[int(result.boxes.cls[i])])
        im = Image.fromarray(im1)
        draw = ImageDraw.Draw(im)
        # print(result.boxes.xyxy[i])
        bbox = result.boxes.xyxy[i]
        x1, y1, x2, y2 = bbox
        # testdraw.rectangle([x1, y1, x2, y2], fill="green")
        # shrink 5%
        testdraw.rectangle([x1, y1, x2, y2], outline="red")
        x1 = x1 + (x2 - x1) * 0.2
        x2 = x2 - (x2 - x1) * 0.2
        y1 = y1 + (y2 - y1) * 0.2
        y2 = y2 - (y2 - y1) * 0.2
        draw.rectangle([x1, y1, x2, y2], fill="white")
        # font = ImageFont.truetype("arial.ttf", 30)
        # testdraw.text(
        #     (x1, y1),
        #     str(i) + names[int(result.boxes.cls[i])],
        #     fill="black",
        #     font=font,

        # )

        # testdraw.rectangle([x1, y1, x2, y2], fill="yellow")
        # draw.rectangle(, fill="white")

        im.save(file.replace(".png", f"_{i}.png"))
        import line

        lines = line.getLines(file.replace(".png", f"_{i}.png"))
        usefulines = []

        if lines is not None:
            for line in lines:
                # print(line)

                # print(line)
                x1, y1, x2, y2 = line[0]
                # testdraw.line([x1, y1, x2, y2], fill="green", width=1)
                # x1, y1 in the boxes.xyxy[i]
                # x2, y2 in the boxes.xyxy[i]
                flag = 0
                if (
                    x1 >= result.boxes.xyxy[i][0]
                    and x1 <= result.boxes.xyxy[i][2]
                    and y1 >= result.boxes.xyxy[i][1]
                    and y1 <= result.boxes.xyxy[i][3]
                ):
                    flag = flag + 1
                if (
                    x2 >= result.boxes.xyxy[i][0]
                    and x2 <= result.boxes.xyxy[i][2]
                    and y2 >= result.boxes.xyxy[i][1]
                    and y2 <= result.boxes.xyxy[i][3]
                ):
                    flag = flag + 1
                if flag == 1:
                    usefulines.append(line[0])

        # print(usefulines, "I")

        for line in usefulines:
            x1, y1, x2, y2 = line
            # testdraw.line([x1, y1, x2, y2], fill="red", width=1)
            for j in con:
                if j == i:
                    continue
                if (
                    x1 >= con[j][0]
                    and x1 <= con[j][2]
                    and y1 >= con[j][1]
                    and y1 <= con[j][3]
                ):
                    print("line", line)
                    testdraw.line([x1, y1, x2, y2], fill="red", width=5)
                    print(i, j)
                    print(map_id_to_name[i], map_id_to_name[j])
                    con_set.add((min(i, j), max(i, j)))
                    print(con_set)
                if (
                    x2 >= con[j][0]
                    and x2 <= con[j][2]
                    and y2 >= con[j][1]
                    and y2 <= con[j][3]
                ):
                    print("line", line)
                    testdraw.line([x1, y1, x2, y2], fill="red", width=5)
                    print(i, j)
                    print(map_id_to_name[i], map_id_to_name[j])
                    con_set.add((min(i, j), max(i, j)))
                    print(con_set)

    imopen.save(file+"_detected.png")
    print(con_set)
    # print(map_id_to_text)

    final_con = []
    final_con=list(con_set)
    # print(map_id_to_text)

    # for i in con_set:
    #     print(i)
    #     # print(map_id_to_name[i[0]])
    #     # print(map_id_to_name[i[1]])
    #     print("-------")
    #     if map_id_to_text[i[0]] == [] and map_id_to_text[i[1]] == []:
    #         print(map_id_to_name[i[0]], map_id_to_name[i[1]])
    #         final_con.append((i[0], i[1]))
    #     else:
    #         continue

    print(final_con)

    import connect_new

    myans = connect_new.connect(final_con, map_id_to_name)

    f = open("results/" + str(rawfile) + ".txt", "w")
    f.write(myans)

    # x = json.load(open(file.replace("jpg", "json")))
    # truth = x["shapes"]
    # flag = False
    # total += len(truth)
    # # total+=length
    # for i in truth:
    #     middle = [
    #         (i["points"][0][0] + i["points"][1][0]) / 2,
    #         (i["points"][0][1] + i["points"][1][1]) / 2,
    #     ]
    #     found = False

    #     for j in range(length):
    #         if (
    #             middle[0] <= result.boxes.xyxy[j][2]
    #             and middle[0] >= result.boxes.xyxy[j][0]
    #             and middle[1] <= result.boxes.xyxy[j][3]
    #             and middle[1] >= result.boxes.xyxy[j][1]
    #         ):
    #             if names[int(result.boxes.cls[j])] == i["label"]:

    #                 correct += 1
    #                 found = True
    #                 break
    #     if not found:
    #         print("not found", i["label"])
    #         if file not in id:
    #             id[file] = 1
    #         else:
    #             id[file] += 1


# print(total)
# print(correct)
# print(correct / total)
# print(id)
