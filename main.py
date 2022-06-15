from fileinput import FileInput
from unicodedata import category
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image



import time
import tensorflow as tf
import numpy as np
import cv2
import os
import cvlib as cv
import pandas as pd







def getOutputsNames(net):
	# Get the names of all the layers in the network
	layersNames = net.getLayerNames()
	# Get the names of the output layers, i.e. the layers with unconnected outputs
	return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


directory = os.getcwd()
PATH_TO_WEIGHTS = directory + "\darknet\cfg\yolov3.weights"
PATH_TO_CFG = directory + "\darknet\cfg\yolov3.cfg"
PATH_TO_COCO_NAMES = directory + "\darknet\data\coco.names"
PATH_TO_LABELS = './annotations/label_map.pbtxt'
PATH_TO_SAVED_MODEL = './exported-models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8' + "/saved_model"
PATH_TO_TEST_IMAGES = directory + "\images\\test\\"

net = cv2.dnn.readNet(PATH_TO_WEIGHTS, PATH_TO_CFG)

# load gender detection model
print('\n\nLoading Gender model...', end='')
start_time = time.time()
model = load_model('exported_model_gender/')
end_time = time.time()
elapsed_time = end_time - start_time
print('\n\n******* Done! Took {} seconds'.format(elapsed_time)+' *********')

gen_classes = ['man','woman']
directory = os.getcwd()
persons_gender = {}

#Reading csv file with pandas and giving names to each column
index=["color","color_name","hex","R","G","B"]
csv = pd.read_csv('colors.csv', names=index, header=None)


# load dress detection model
print('\n\nLoading Dress model...', end='')
start_time = time.time()
dress_model = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('\n\n******* Done! Took {} seconds'.format(elapsed_time)+' *********')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

		
classes = []
persons = []
male_count = 0
female_count = 0
x = y = w = h = text = None
COLORS = None
with open(PATH_TO_COCO_NAMES, 'r') as f:
	classes = f.read().splitlines()
# initialize a list of colors to represent each possible class label
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")


# video Capture
# cap = cv2.VideoCapture(directory + '\\videos\\video.mp4')

def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    keypoint_scores=None,
    keypoint_edges=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    mask_alpha=.4,
    groundtruth_box_visualization_color='black',
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a uint8 numpy array of shape [N, image_height, image_width],
      can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None.
    keypoint_scores: a numpy array of shape [N, num_keypoints], can be None.
    keypoint_edges: A list of tuples with keypoint indices that specify which
      keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
      edges from keypoint 0 to 1 and from keypoint 2 to 4.
    track_ids: a numpy array of shape [N] with unique track ids. If provided,
      color-coding of boxes will be determined by these ids, and not the class
      indices.
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box or keypoint to be
      visualized.
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    mask_alpha: transparency value between 0 and 1 (default: 0.4).
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_boxes: whether to skip the drawing of bounding boxes.
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
    skip_track_ids: whether to skip track id when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = viz_utils.collections.defaultdict(list)
  box_to_color_map = viz_utils.collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = viz_utils.collections.defaultdict(list)
  box_to_keypoint_scores_map = viz_utils.collections.defaultdict(list)
  box_to_track_ids_map = {}
  final_label = None
  final_bounding_box_coordinates = None
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(boxes.shape[0]):
    if max_boxes_to_draw == len(box_to_color_map):
      break
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if keypoint_scores is not None:
        box_to_keypoint_scores_map[box].extend(keypoint_scores[i])
      if track_ids is not None:
        box_to_track_ids_map[box] = track_ids[i]
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
            if not agnostic_mode:
                if classes[i] in viz_utils.six.viewkeys(category_index):
                    class_name = category_index[classes[i]]['name']
                    print(class_name)
                else:
                    class_name = 'N/A'
            display_str = str(class_name)
        final_label = display_str
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(round(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, round(100*scores[i]))
        if not skip_track_ids and track_ids is not None:
          if not display_str:
            display_str = 'ID {}'.format(track_ids[i])
          else:
            display_str = '{}: ID {}'.format(display_str, track_ids[i])
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        elif track_ids is not None:
          prime_multipler = viz_utils._get_multiplier_for_color_randomness()
          box_to_color_map[box] = viz_utils.STANDARD_COLORS[
              (prime_multipler * track_ids[i]) % len(viz_utils.STANDARD_COLORS)]
        else:
          box_to_color_map[box] = viz_utils.STANDARD_COLORS[
              classes[i] % len(viz_utils.STANDARD_COLORS)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if instance_masks is not None:
      viz_utils.draw_mask_on_image_array(
          image,
          box_to_instance_masks_map[box],
          color=color,
          alpha=mask_alpha
      )
    if instance_boundaries is not None:
      viz_utils.draw_mask_on_image_array(
          image,
          box_to_instance_boundaries_map[box],
          color='red',
          alpha=1.0
      )
    viz_utils.draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=0 if skip_boxes else line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    if keypoints is not None:
      keypoint_scores_for_box = None
      if box_to_keypoint_scores_map:
        keypoint_scores_for_box = box_to_keypoint_scores_map[box]
      viz_utils.draw_keypoints_on_image_array(
          image,
          box_to_keypoints_map[box],
          keypoint_scores_for_box,
          min_score_thresh=min_score_thresh,
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates,
          keypoint_edges=keypoint_edges,
          keypoint_edge_color=color,
          keypoint_edge_width=line_thickness // 2)

    final_bounding_box_coordinates = [xmin, ymin, xmax, ymax]
  return final_label, final_bounding_box_coordinates


#function to calculate minimum distance from all colors and get the most matching color
def getColorName(R,G,B, csv):
  minimum = 10000
  cname = None
  for i in range(len(csv)):
    d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
    if(d<=minimum):
      minimum = d
      cname = csv.loc[i,"color_name"]
  return cname



def genderDetection(img, text):
  pass


# Webcam
def webcam():
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        H, W, _ = img.shape


        # 
        #       :) :)
        # 
        # PersonDetection      -------
        #                            |               
        #                            V           
        #  

        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = getOutputsNames(net)
        layerOutputs = net.forward(output_layers_names)
        boxes = []
        confidences = []
        class_ids = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # print(len(boxes))
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # ensure at least one detection exists
        if len(indexes) > 0:
            # loop over the indexes we are keeping
            for i in indexes.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                if classes[class_ids[i]] in ["person"]: 
                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in COLORS[class_ids[i]]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    if classes[class_ids[i]] == "person":
                        persons.append(img[y: y + h, x: x + w])
                        text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
                        # cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 
        #       :) :)
        # 
        # GenderDetection      -------
        #                            |               
        #                            V           
        #  


        # apply face detection
        face, confidence = cv.detect_face(img)

        # loop through detected faces
        for idx, f in enumerate(face):
            # get corner points of face rectangle        
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # draw rectangle over face
            # cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)

            # crop the detected face region
            face_crop = np.copy(img[startY:endY,startX:endX])
            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue

            # preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (96,96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # apply gender detection on face
            conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

            # get label with max accuracy
            idx = np.argmax(conf)
            label = gen_classes[idx]
            label = "{}: {:.2f}%".format(label, conf[idx] * 100)
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            # if label == "man":
            #     male_count = male_count + 1
            # else:
            #     female_count = female_count + 1
            # write label and confidence above face rectangle
            # cv2.putText(img, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
            cv2.putText(img, text + " : " + label , (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # cv2.putText(img,label , (x , y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



        # 
        #       :) :)
        # 
        # DressDetection      -------
        #                            |               
        #                            V           
        #  

        image_np = img
        # print(type(img))
        # print(img)
        # image_np = self.frame_into_numpy_array(img)

        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #    np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = dress_model(input_tensor)

        # detection_classes should be ints.
        image_np_with_detections = image_np.copy()

        # The following processing is only for single image
        detection_boxes = tf.squeeze(detections['detection_boxes'], [0])
        detection_masks = tf.squeeze(detections['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(detections['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0],[real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        
        # Follow the convention by adding back the batch dimension
        detections['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

        detections['num_detections'] = int(detections['num_detections'][0])
        detections['detection_classes'] = detections['detection_classes'][0].numpy().astype(np.uint8)
        detections['detection_boxes'] = detections['detection_boxes'][0].numpy()
        detections['detection_scores'] = detections['detection_scores'][0].numpy()
        detections['detection_masks'] = detections['detection_masks'][0].numpy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            instance_masks=detections.get('detection_masks'),
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30)

                

        # display output
        cv2.imshow("Detection", image_np_with_detections)

        key = cv2.waitKey(1)
        if key==27:
            # print("Total Persons Detected   : " + str(male_count + female_count))
            # print("Male Identified          : " + str(male_count))
            # print("Female Identified        : " + str(female_count))
            break


# Image file
def imageFile(imgPath, shirt_type, gender):
    imgPath = PATH_TO_TEST_IMAGES + imgPath
    print(imgPath)
    img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
    H, W, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = getOutputsNames(net)
    layerOutputs = net.forward(output_layers_names)
    boxes = []
    confidences = []
    class_ids = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
			    # scale the bounding box coordinates back relative to the
			    # size of the image, keeping in mind that YOLO actually
			    # returns the center (x, y)-coordinates of the bounding
			    # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
			    # update our list of bounding box coordinates, confidences,
			    # and class IDs
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

	# print(len(boxes))
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	 # ensure at least one detection exists
    if len(indexes) > 0:
		# loop over the indexes we are keeping
        for i in indexes.flatten():
			# extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if classes[class_ids[i]] in ["person"]: 
			    # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[class_ids[i]]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                if classes[class_ids[i]] == "person":
                    persons.append(img[y: y + h, x: x + w])
                    text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
                    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 
    #       :) :)
    # 
    # GenderDetection      -------
    #                            |               
    #                            V           
    #  


    # apply face detection
    face, confidence = cv.detect_face(img)

    # loop through detected faces
    for idx, f in enumerate(face):
        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        # cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(img[startY:endY,startX:endX])
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx = np.argmax(conf)
        label = gen_classes[idx]
        gend = label
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        # if label == "man":
        #     male_count = male_count + 1
        # else:
        #     female_count = female_count + 1
        # write label and confidence above face rectangle
        # cv2.putText(img, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        # cv2.putText(img, text + " : " + label , (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # cv2.putText(img, text + " : " + label , (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        text += " : " + label
        cv2.putText(img,label , (x , y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



    # 
    #       :) :)
    # 
    # DressDetection      -------
    #                            |               
    #                            V           
    #  

    image_np = img
    # print(type(img))
    # print(img)
    # image_np = self.frame_into_numpy_array(img)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #    np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = dress_model(input_tensor)

    # detection_classes should be ints.
    image_np_with_detections = image_np.copy()

    # The following processing is only for single image
    detection_boxes = tf.squeeze(detections['detection_boxes'], [0])
    detection_masks = tf.squeeze(detections['detection_masks'], [0])
    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    real_num_detection = tf.cast(detections['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0],[real_num_detection, -1, -1])
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    
    # Follow the convention by adding back the batch dimension
    detections['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

    detections['num_detections'] = int(detections['num_detections'][0])
    detections['detection_classes'] = detections['detection_classes'][0].numpy().astype(np.uint8)
    detections['detection_boxes'] = detections['detection_boxes'][0].numpy()
    detections['detection_scores'] = detections['detection_scores'][0].numpy()
    detections['detection_masks'] = detections['detection_masks'][0].numpy()

    final_dress_label, final_bounding_box_coordinates = visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        instance_masks=detections.get('detection_masks'),
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30)

    colour_name = "Not Detected"
    if final_bounding_box_coordinates is not None:
      xmin = int(final_bounding_box_coordinates[0] * W)
      ymin = int(final_bounding_box_coordinates[1] * H)
      xmax = int(final_bounding_box_coordinates[2] * W)
      ymax = int(final_bounding_box_coordinates[3] * H)
      B,G,R = img[int(xmax/2),int(ymax/2)]
      print(B,G,R)
      colour_name = getColorName(R, G, B, csv)
      print(colour_name)

    final_bounding_box_color = None
    if final_dress_label == shirt_type and gender == gend:
        text += " :: Dress Code : Yes :: Color : " + colour_name
        # BGR
        final_bounding_box_color = (0, 255, 0)
    else:
        text += " :: Dress Code : No :: Color : " + colour_name
        # BGR
        final_bounding_box_color = (0, 0, 255)

    print(text)
    # cv2.imwrite("shirt.jpg", image_np_with_detections[ymin:ymax, xmin:xmax])

    cv2.putText(image_np_with_detections, text , (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, final_bounding_box_color, 2)  
    cv2.rectangle(image_np_with_detections, (x, y), (x + w, y + h), final_bounding_box_color, 2)
    # print(category_index)
    # print(final_dress_label)
    # print([category_index.get(value) for index,value in enumerate(classes[0]) if scores[index] > 0.5])   



    # display output
    cv2.imshow("Detection", image_np_with_detections)
    cv2.waitKey(0)
    # release resources
    cv2.destroyAllWindows()





if __name__ == "__main__":

    # print("Enter shirt type : ")
    # shirt_type = input()
    shirt_type = "shirt"

    # print("Enter gender : ")
    # gender = input()
    gender = "man"

    # Give image name from images directory
    imageFile('6.jpg', shirt_type, gender)

    # Webcam
    # webcam()

