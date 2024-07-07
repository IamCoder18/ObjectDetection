import time
import base64
import numpy as np
from flask import Flask, request
from flask_cors import CORS
from sahi.utils.cv import (get_bool_mask_from_coco_segmentation, read_image_as_pil)
from ultralyticsplus import YOLO

app = Flask(__name__)
CORS(app)

# model = YOLO('mshamrai/yolov8n-visdrone')
model = YOLO('techzizou/yolov8n')
model.overrides['conf'] = 0.25
model.overrides['iou'] = 0.45
model.overrides['agnostic_nms'] = False
model.overrides['max_det'] = 1000


@app.route("/", methods=['POST'])
def upload_image():
    start_time = time.time()

    data = request.json
    head, data = str(data.get('filename')).split(",", 1)

    mime_type = head.split(";")[0].split("/")[1]
    filename = f"image.{mime_type}"

    decoded_data = base64.b64decode(data)

    with open(filename, 'wb') as f:
        f.write(decoded_data)

    print(f"Data URI saved as: {filename}")

    results = model.predict(filename)

    image = read_image_as_pil(filename)
    np_image = np.ascontiguousarray(image)

    names = model.model.names

    masks = results[0].masks
    boxes = results[0].boxes

    object_predictions = []
    if boxes is not None:
        det_ind = 0
        for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
            if masks:
                img_height = np_image.shape[0]
                img_width = np_image.shape[1]
                segments = masks.segments
                segments = segments[det_ind]
                segments[:, 0] = segments[:, 0] * img_width
                segments[:, 1] = segments[:, 1] * img_height
                segmentation = [segments.ravel().tolist()]

                bool_mask = get_bool_mask_from_coco_segmentation(
                    segmentation, width=img_width, height=img_height
                )
                if sum(sum(bool_mask == 1)) <= 2:
                    continue
                object_predictions.append([[img_height, img_width], names[int(cls)], round(conf.item() * 100)])
            else:
                object_predictions.append([xyxy.tolist(), names[int(cls)], round(conf.item() * 100)])
            det_ind += 1

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Detection time: {execution_time:.2f} seconds")
    return {"result": object_predictions}, 200


@app.route("/activate", methods=['GET'])
def activate():
    return "Ok", 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
