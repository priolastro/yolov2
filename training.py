import glob
from library import parse_annotations, ground_truth_generator, yolo_loss, get_classes, get_x_train, read_anchors
import tensorflow as tf
from keras.layers import Lambda, Input, Layer
from tensorflow.keras.models import Model
import numpy as np
from datetime import datetime
from keras.layers import Input, Lambda, Conv2D
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2 as cv
import os


# create custom layer for computing the loss
class YoloLossLayer(Layer):
    def __init__(self, anchors, num_classes, **kwargs):
        super(YoloLossLayer, self).__init__(**kwargs)
        self.anchors = anchors
        self.num_classes = num_classes

    def call(self, inputs):
        yolo_model_output, boxes_input, detectors_mask_input, matching_boxes_input = inputs
        loss = yolo_loss([yolo_model_output, boxes_input, detectors_mask_input, matching_boxes_input], self.anchors, self.num_classes)
        self.add_loss(loss, inputs=inputs)
        return yolo_model_output  # Return the output for model construction
IMAGES_FOLDER = X:/object_detection_dataset/images/
ANNOTATIONS_FOLDER = X:/object_detection_dataset/labels/*
LABELS = get_classes("./model_data/coco_classes.txt")
ANCHORS = read_anchors("./model_data/yolo_anchors.txt")
IMAGE_H, IMAGE_W = 416,416
GRID_H,  GRID_W  = int(IMAGE_H/32), int(IMAGE_W/32) # GRID size = IMAGE size / 32
CLASS            = len(LABELS)
TRAIN_BATCH_SIZE = 32


def main():
    images_paths_train = sorted(glob.glob(IMAGES_FOLDER+"/*.jpeg"))
    annotations_paths_train = sorted(glob.glob(ANNOTATIONS_FOLDER+"/*.txt"))

    print("Total number of images: {}\nTotal number of annotations: {}".format(len(images_paths_train), len(annotations_paths_train)))

    true_boxes_train = parse_annotations(annotations_paths_train)
    print("NUMBER OF IMAGES, MAX BOX PER IMAGE, BOX DIMENSION (class, x, y, w, h)", true_boxes_train.shape)

    x_train = get_x_train(images_paths_train, image_shape=(IMAGE_W, IMAGE_W))
    data_set_train = tf.data.Dataset.from_tensor_slices((x_train, true_boxes_train))
    data_set_train = data_set_train.batch(TRAIN_BATCH_SIZE)
    train_gen = ground_truth_generator(data_set_train, ANCHORS, GRID_H, GRID_W, CLASS)
    # Loading the pretrained yolo v2
    # yolo_model = tf.keras.models.load_model("model_data/", compile=False)

    # Loading conveerted tiny yolo weights
    yolo_model = tf.keras.models.load_model("./model_data/yolov2-tiny.h5", compile=False)
    print(type(yolo_model))
    # yolo_model.summary()


    # Loading the pretrained yolo v2
    # yolo_model = tf.keras.models.load_model("model_data/", compile=False)

    # Loading conveerted tiny yolo weights
    yolo_model = tf.keras.models.load_model("./model_data/yolov2-tiny.h5", compile=False)
    print(type(yolo_model))
    # yolo_model.summary()

    # Modification in order to change numer of classes and apply transfer learning
    topless_layers = yolo_model.layers[:-1] # remove old prediction layer
    new_yolo_model = tf.keras.models.Model(inputs=yolo_model.input, outputs=topless_layers[-1].output) # create medusa model (without last layer)
    final_layer = Conv2D(len(ANCHORS)*(5+len(LABELS)), (1, 1), activation='linear',  name="Prediction_Layer")(new_yolo_model.output)  # create new layer for prediction with different number of labels
    final_model = tf.keras.models.Model(inputs=new_yolo_model.inputs, outputs=final_layer)  # assemble new model
    final_model.summary()


    # add Lambda layer to compute loss and connect it to new inputs for boxes, detectors masks and matching_boxes masks 

    # create new input layers
    boxes_input = Input(shape=(None, 5),name="Boxes")
    detectors_mask_input = Input(shape=(GRID_H,GRID_W,5,1), name="detectors_mask")
    matching_boxes_input = Input(shape=(GRID_H,GRID_W,5,5),name="matching_boxes_input")


    # Instantiate the custom loss layer and select the right number of classes you want to predict
    print("Number of classes: {}".format(CLASS))
    yolo_loss_layer = YoloLossLayer(anchors=ANCHORS, num_classes=CLASS)

    # Connect the new inputs to the loss layer
    model_output = yolo_loss_layer([final_model.output, boxes_input, detectors_mask_input, matching_boxes_input])


    # Create the trainable new YOLO model having a loss layer to compute the loss during training plus the suitable inputs defined above
    yolo_model_with_loss = Model(inputs=[final_model.input, boxes_input, detectors_mask_input, matching_boxes_input], outputs=model_output)
    # Compile the model with a dummy optimizer and loss (since the loss is already added within the custom layer)
    base_learning_rate = 1e-3
    yolo_model_with_loss.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), loss=None)
    print("Number of layers in the base model: ", len(yolo_model_with_loss.layers))

    # for layer in yolo_model_with_loss.layers[:74]:
    #     layer.trainable = False

    # Display the model summary 
    # Check number of weights. Must be equal to original yolo_v2 imported at start because neither loss layer or input layers introduce any new weigths to optimize
    yolo_model_with_loss.summary(show_trainable=True)

    # Freeze all layers not needed for transfer learning 
    print("Number of layers in the base model: ", len(yolo_model_with_loss.layers))
    # Freeze_up_to = 74  # this to train only last prediction layer full yolo v2
    # Freeze_up_to = 31  # this to train only last prediction layer
    # Freeze_up_to = 28  # this to train last convolutional layer
    Freeze_up_to = 25 # this to train second-last convolutional layer

    for layer in yolo_model_with_loss.layers[:Freeze_up_to]:
        layer.trainable=False
    yolo_model_with_loss.summary(show_trainable=True)
    img, boxes, detectors_mask,matching_true_boxes = train_gen
    print(img.shape)
    # Train the model
    initial_epochs = 100
    history = yolo_model_with_loss.fit([img, boxes, detectors_mask, matching_true_boxes],np.zeros(len(img)),batch_size=1, epochs=initial_epochs)
    # fine tuning makeing trainable also the second last convolutional layer
    # fine_tune_at = 28 # this to train last convolutional layer
    # fine_tune_at = 25 # this to train second-last convolutional layer
    fine_tune_at = 22 # this to train third-last convolutional layer

    print("Number of layers in the main model: ", len(yolo_model_with_loss.layers))
    print("Fine tune at: ", fine_tune_at)
    for layer in yolo_model_with_loss.layers:
        layer.trainable = True
    for layer in yolo_model_with_loss.layers[:fine_tune_at]:
        layer.trainable = False
    yolo_model_with_loss.summary(show_trainable=True)

    metrics = ['accuracy']
    yolo_model_with_loss.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate*0.01), loss=None)


    fine_tune_epochs = 50
    total_epochs =  initial_epochs + fine_tune_epochs

    history_fine = yolo_model_with_loss.fit([img, boxes, detectors_mask, matching_true_boxes],np.zeros(len(img)),batch_size=1, epochs=total_epochs, initial_epoch=history.epoch[-1])

    model2= Model(inputs=yolo_model_with_loss.input[0], outputs=yolo_model_with_loss.layers[-5].output)
    model2.summary()
    name_model = 'yolo_model_'+ str(datetime.today().strftime ('%d-%b-%Y-%H%M%S'))
    save_as = os.path.join("./saved_models", name_model)
    print("saving model as ", save_as)
    model2.save(save_as)
    converter = tf.lite.TFLiteConverter.from_saved_model(save_as)
    tflite_model = converter.convert()
    tflite_model_name = os.path.join(save_as,name_model+'.tflite') 
    print("saving model as ", tflite_model_name)
    with open(tflite_model_name, 'wb') as f:
        f.write(tflite_model)

        
if __name__ == "__main__":
    main()

























