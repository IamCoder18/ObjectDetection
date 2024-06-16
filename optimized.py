import cv2
from collections import Counter
from inflect import engine
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch


def capture():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening camera")
        exit()

    num_frames = 30
    captured_frame = None

    for i in range(num_frames):
        ret, frame = cap.read()

        if not ret:
            print("Could not capture frame")
            continue

        if captured_frame is None:
            captured_frame = frame

    cap.release()

    if captured_frame is not None:
        cv2.imwrite('image.jpg', captured_frame)
        height, width, _ = captured_frame.shape
    else:
        print("Failed to capture image")


def detect(url):
    image = Image.open(url)

    # you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    objects = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        objects.append(model.config.id2label[label.item()])
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    return objects


def count_and_format(obj_list):
    p = engine()
    counts = Counter(obj_list)
    itemsList = []
    for item, count in counts.items():
        if count == 1:
            itemsList.append(p.a(item))
        else:
            itemsList.append(str(count) + " " + p.plural(item))
    return p.join(itemsList)


capture()
