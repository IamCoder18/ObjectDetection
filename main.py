import cv2
from collections import Counter
from inflect import engine
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf


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

        # Calculate section width (assuming integer division for equal sections)
        section_width = width // 3

        # Split image into sections
        sections = [captured_frame[:, i * section_width: (i + 1) * section_width] for i in range(3)]

        # Define base filename and extension (modify as needed)
        base_filename = "section"
        extension = ".jpg"

        # Save each section to a separate file
        for i, section in enumerate(sections):
            filename = f"{base_filename}{i + 1}{extension}"
            cv2.imwrite(filename, section)
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


print("Done Importing")
print("Capturing Images")
capture()
print("Done Capturing Images")

text = ""

print("Detecting From Image 1")
objectL = detect("./section1.jpg")
print(objectL)
print("Done Detecting From Image 1")

if len(objectL) > 0:
    print("Formatting")
    textL = ""
    textO = count_and_format(objectL)
    if textO[0] != "a":
        textL += "On the left, there are " + textO + ". "
    else:
        textL += "On the left, there is a " + textO + ". "
    print(textL)
    text += textL
    print("Done Formatting")

print("Detecting From Image 2")
objectM = detect("./section2.jpg")
print(objectM)
print("Done Detecting From Image 2")

if len(objectM) > 0:
    print("Formatting")
    textM = ""
    textO = count_and_format(objectM)
    if textO[0] != "a":
        textM += "In the middle, there are " + textO + ". "
    else:
        textM += "In the middle, there is a " + textO + ". "
    print(textM)
    text += textM
    print("Done Formatting")

print("Detecting From Image 3")
objectR = detect("./section3.jpg")
print(objectR)
print("Done Detecting From Image 3")

if len(objectR) > 0:
    print("Formatting")
    textR = ""
    textO = count_and_format(objectR)
    if textO[0] != "a":
        textR += "On the right, there are " + textO + ". "
    else:
        textR += "On the right, there is " + textO + ". "
    print(textR)
    text += textR
    print("Done Formatting")

print(text)

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
