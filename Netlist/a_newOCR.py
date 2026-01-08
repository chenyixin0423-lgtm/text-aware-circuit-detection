from PIL import Image
import easyocr
def inpaint_text(imagePath, pipeline):
    import cv2
    import numpy as np
    import easyocr
    results = pipeline.readtext(imagePath)
    imagep = Image.open(imagePath)
    width, height = imagep.size
    imagep.close()
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in results:
        polygon =  np.array(box[0],np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [polygon], 255)
    image = cv2.imread(imagePath)
    inpainted_img = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
    return inpainted_img

class TextRemover:
    def __init__(self):
        self.pipeline = easyocr.Reader(["en"])
    def remove_text(self, img_path):        
        img_text_removed = inpaint_text(img_path, self.pipeline)
        return img_text_removed