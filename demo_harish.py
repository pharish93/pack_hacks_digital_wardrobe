import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from run_network import run_inference_for_single_image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

import glob
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = []
a = str(PATH_TO_TEST_IMAGES_DIR+"/*.jpg")
for file in glob.glob(a):
    TEST_IMAGE_PATHS.append(file)

IMAGE_SIZE = (12, 8)

def create_sub_images(image_np,detection_boxes,detection_classes,detection_scores,
        category_index,detection_masks,index):
    min_score_thresh = 0.7
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')

    for i in range(0,detection_boxes.shape[0]):
        if (category_index[detection_classes[i]]['name'] == 'person') and detection_scores[i] > min_score_thresh:
            box = tuple(detection_boxes[i].tolist())
            mask = detection_masks[i]
            ymin, xmin, ymax, xmax = box
            im_width, im_height = image_pil.size
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                                          ymin * im_height, ymax * im_height)
            if left < 0: left = 0
            if right < 0 : right = 0
            if top <0 : top =0
            if bottom <0 : bottom = 0
            if left >= im_width : left = im_width -1
            if right >= im_width: right = im_width - 1
            if top >= im_height: top = im_height - 1
            if bottom >= im_height: bottom = im_height - 1

            img_temp = image_np[int(top):int(bottom), int(left):int(right)]
            mask_temp = mask[int(top):int(bottom), int(left):int(right)]
            idx = (mask_temp != 0)
            img_new = np.zeros(img_temp.shape,dtype=np.float32)
            img_new[idx] = img_temp[idx]
            img_new = img_new[...,::-1]
            file_name = 'sub_images/person' + str(index)+str(i) + '.jpg'
            cv2.imwrite(file_name, img_new)


            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            # gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
            gray = cv2.imread(file_name,0)
            img_new = cv2.imread(file_name)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img_new, (x, y), (x + w, y + h), (255, 0, 0), 2)
                file_name = 'sub_image_face/person' + str(index) + str(i) + '.jpg'
                cv2.imwrite(file_name, img_new)

                lower = np.array([0, 48, 80], dtype="uint8")
                upper = np.array([20, 255, 255], dtype="uint8")

                converted = cv2.cvtColor(img_new, cv2.COLOR_BGR2HSV)
                skinMask = cv2.inRange(converted, lower, upper)

                # apply a series of erosions and dilations to the mask
                # using an elliptical kernel
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                skinMask = cv2.erode(skinMask, kernel, iterations=2)
                skinMask = cv2.dilate(skinMask, kernel, iterations=2)

                # blur the mask to help remove noise, then apply the
                # mask to the frame
                skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
                skinMask = cv2.bitwise_not(skinMask)
                skin = cv2.bitwise_and(img_new, img_new, mask=skinMask)

                # show the skin in the image along with the mask
                file_name = 'skin/person_skin' + str(index) + str(i) + '.jpg'

                cv2.imwrite(file_name, skin)

    np.copyto(image_np, np.array(image_pil))


for index,image_path in enumerate(TEST_IMAGE_PATHS):
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    # image_np = vis_util.visualize_boxes_and_labels_on_image_array(
    #     image_np,
    #     output_dict['detection_boxes'],
    #     output_dict['detection_classes'],
    #     output_dict['detection_scores'],
    #     category_index,
    #     instance_masks=output_dict.get('detection_masks'),
    #     use_normalized_coordinates=True,
    #     line_thickness=8)
    # plt.imsave('outputs/output'+str(index)+'.jpg',image_np)

    create_sub_images(image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    output_dict.get('detection_masks'),index)



    # plt.figure(figsize=IMAGE_SIZE)

