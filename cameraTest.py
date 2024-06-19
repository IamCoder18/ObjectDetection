import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error opening camera")
    exit()

num_frames = 10
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
    raise SystemError