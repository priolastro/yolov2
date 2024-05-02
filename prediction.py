from tensorflow.keras.models import load_model
from library import preprocess_image_yolo_CV, add_black_padding_cv, preprocess_data_yolo_CV, yolo_head, read_anchors, yolo_eval, get_classes
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import random

def main():
    class_names = get_classes("./model_data/coco_classes.txt")
    anchors = read_anchors("model_data/yolo_anchors.txt")
    # yolo_model = load_model("./model_data/yolov2-tiny.h5", compile=False)
    yolo_model = load_model("./model_data/", compile=False)
    model_image_size = yolo_model.input_shape[1:-1]
    print(model_image_size)
    to_eval_images = glob.glob("X:/object_detection_dataset/images/*")[9:11]
    for path in to_eval_images:
        print(path)
        image = preprocess_image_yolo_CV(path)
        if image.shape[0] != image.shape[1]:
            image = add_black_padding_cv(image)
        image_data = preprocess_data_yolo_CV(image, model_image_size)
        yolo_model_outputs = yolo_model(image_data)
        yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
        out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.shape[1],  image.shape[0]], 10, 0.3, 0.5)
        for box_num, box_cord in enumerate(out_boxes):
            box = [int(element) for element in box_cord]
            class_name = class_names[out_classes[box_num]]
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_thickness = 5
            text_color = (255, box_num, 0)
            text_size = cv.getTextSize(class_name, font, font_scale, font_thickness)[0]
            text_width, text_height = text_size
            text_x = box[3]  
            text_y = box[2] - 5 - text_height
            text_org = (text_x, text_y)
            cv.putText(image, class_name, text_org, font, font_scale, text_color, font_thickness)
            cv.rectangle(image, (box[1],box[0]), (box[3], box[2]), color=(255,box_num,0), thickness=3)
        plt.imshow(image)
        plt.show()
        plt.close()

    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = frame.astype('float32')/255.

        # print(frame.shape)
        if frame.shape[0] != frame.shape[1]:
            frame = add_black_padding_cv(frame)
        image_data = preprocess_data_yolo_CV(frame, model_image_size)
        yolo_model_outputs = yolo_model(image_data)
        yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
        _, out_boxes, out_class = yolo_eval(yolo_outputs, [frame.shape[0],  frame.shape[1]], 10, 0.3, 0.5)
        for box_num, box_cord in enumerate(out_boxes):
            box = [int(element) for element in box_cord]
            cv.rectangle(frame, (box[1], box[0]), (box[3],box[2]), color=(255,box_num,0), thickness=5)
            class_name = class_names[out_class[box_num]]
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_color = (255, 255, 255)
            text_size = cv.getTextSize(class_name, font, font_scale, font_thickness)[0]
            text_width, text_height = text_size
            text_x = box[1]  
            text_y = box[0] - 5 - text_height  
            text_org = (text_x, text_y)
            cv.putText(frame, class_name, text_org, font, font_scale, text_color, font_thickness)

        cv.imshow('Webcam', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()




if __name__ == "__main__":
    main()