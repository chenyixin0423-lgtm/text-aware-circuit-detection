from PIL import Image


def crop(image_path, bbox):
    
    image = Image.open(image_path)

    cropped_image = image.crop(bbox)
    cropped_image.save(f"cropped_image_{bbox}.png")
    return OCR(bbox)


def OCR(bbox):
    import easyocr
    import cv2
    from PIL import ImageDraw

    reader = easyocr.Reader(["en"])

    
    results = reader.readtext(f"cropped_image_{bbox}.png", detail=1,text_threshold=0.5,low_text=0.0) # text_threshold=0.2
    print(results)
    ans = []
    for result in results:
        print(result)
        bbox, text, confidence = result
        print(text)
        print(result)
        ans.append(text)
    return ans
    #     top_left = tuple(map(int, bbox[0]))

    #     bottom_right = tuple(map(int, bbox[2]))
    #     x1 = top_left[0]
    #     y1 = top_left[1]
    #     x2 = bottom_right[0]
    #     y2 = bottom_right[1]
    #     draw.rectangle([x1, y1, x2, y2], fill="white")

    # return image1
