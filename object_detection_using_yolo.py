"""
Module 1	- Person Detection
Date		- 16/01/2022
"""
"""**Actual Code For Human Detection using YOLOv3**"""


#from google.colab.patches import cv2_imshow
import cv2
import os
import numpy as np


class HumanDetection:
	def __init__(self):
		self.directory = os.getcwd()
		self.PATH_TO_WEIGHTS = self.directory + "\darknet\cfg\yolov3.weights"
		self.PATH_TO_CFG = self. directory + "\darknet\cfg\yolov3.cfg"
		self.PATH_TO_COCO_NAMES = self.directory + "\darknet\data\coco.names"
		self.net = cv2.dnn.readNet(self.PATH_TO_WEIGHTS, self.PATH_TO_CFG)
		
		self.classes = []
		self.persons = []
		self.COLORS = None
		with open(self.PATH_TO_COCO_NAMES, 'r') as f:
			self.classes = f.read().splitlines()
		# initialize a list of colors to represent each possible class label
		self.COLORS = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")		
		
	def getOutputsNames(self, net):
		# Get the names of all the layers in the network
		layersNames = net.getLayerNames()
		# Get the names of the output layers, i.e. the layers with unconnected outputs
		return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

	def videoCaptureFromCamera(self):
		# video Capture
		cap = cv2.VideoCapture(0)
		while True:
			_, img = cap.read()
			H, W, _ = img.shape
			blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
			self.net.setInput(blob)
			output_layers_names = self.getOutputsNames(self.net)
			layerOutputs = self.net.forward(output_layers_names)
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
					if self.classes[class_ids[i]] in ["person"]:
						# draw a bounding box rectangle and label on the image
						color = [int(c) for c in self.COLORS[class_ids[i]]]
						cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
						if self.classes[class_ids[i]] == "person":
							self.persons.append(img[y: y + h, x: x + w])
						text = "{}: {:.4f}".format(self.classes[class_ids[i]], confidences[i])
						cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
						
			cv2.imshow("Webcam",img)
			key = cv2.waitKey(1)
			if key==27:
				break
		# cv2.imwrite("Person.jpg", self.persons[0])
		cap.release()
		cv2.destroyAllWindows()

	def videoCaptureFromFile(self):
		#video Capture
		video_path = self.directory + "\\videos\\vtest.mp4"
		print(video_path)
		cap = cv2.VideoCapture(video_path)
		while True:
			_, img = cap.read()
			H, W, _ = img.shape
			blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
			self.net.setInput(blob)
			output_layers_names = self.getOutputsNames(self.net)
			layerOutputs = self.net.forward(output_layers_names)
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
					if self.classes[class_ids[i]] in ["person"]:
						# draw a bounding box rectangle and label on the image
						color = [int(c) for c in self.COLORS[class_ids[i]]]
						cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
						if self.classes[class_ids[i]] == "person":
							self.persons.append(img[y: y + h, x: x + w])
						text = "{}: {:.4f}".format(self.classes[class_ids[i]], confidences[i])
						cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
						
			cv2.imshow("Video",img)
			key = cv2.waitKey(1)
			if key==27:
				break
		cap.release()
		cv2.destroyAllWindows()

	def imageFile(self, image):
		img_path = self.directory + "\images\\"+ image
		print(img_path)
		img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
		H, W, _ = img.shape

		blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
		self.net.setInput(blob)
		output_layers_names = self.getOutputsNames(self.net)
		layerOutputs = self.net.forward(output_layers_names)

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
					if self.classes[class_ids[i]] in ["person"]:
						# draw a bounding box rectangle and label on the image
						color = [int(c) for c in self.COLORS[class_ids[i]]]
						cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
						if self.classes[class_ids[i]] == "person":
							self.persons.append(img[y: y + h, x: x + w])
						text = "{}: {:.4f}".format(self.classes[class_ids[i]], confidences[i])
						cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
						
			cv2.imshow("Webcam",img)
		
		cv2.imshow('Person', self.persons[0])
		# print(len(self.persons))
		cv2.waitKey(0)
		cv2.destroyAllWindows()



# """  Driver Code Or Main  """
if __name__ == "__main__":
	humanDetection = HumanDetection()
	
	# Give image name from images directory
	# humanDetection.imageFile('img10.jpg')

	# Press Esc to exit from loop
	humanDetection.videoCaptureFromCamera()

	# Not working
	# humanDetection.videoCaptureFromFile()
