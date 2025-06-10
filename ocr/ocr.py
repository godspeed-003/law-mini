import os
from os import listdir
from os.path import isfile, join
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"D:\gdgsc\ocr\vision_key.json"
from google.cloud import vision
from my_timer import my_timer
import time

def detect_text(path):
    client = vision.ImageAnnotatorClient()
    with open(path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    ocr_text = []
    for text in texts:
        ocr_text.append(f"\r\n{text.description}")
    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    return texts[0].description if texts else "No text detected"

@my_timer
def main():
    mypath = r"D:\gdgsc\ocr\image" # path to the image folder
    only_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for image_path in only_files:
        full_image_path = os.path.join(mypath, image_path)
        text = detect_text(full_image_path)
        print(image_path)
        print(text)

if __name__ == "__main__":
     main()
