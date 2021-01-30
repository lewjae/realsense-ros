#!/usr/bin/env python
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import collections
#import statistics
import math
import tarfile
import os.path
import numpy as np

#from threading import Lock, Thread
from time import sleep

#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

# ROS related imports
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import String , Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
#from pincair_msgs.msg import DetectedBox

# ZED imports
#import pyzed.sl as sl
sys.path.append('utils')

# ## Object detection imports
from object_detection.utils import ops as utils_ops

from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.builders import model_builder

print("Tensorflow version: ", tf.__version__)

# This main thread will run the object detection, the capture thread is loaded later
DATA_DIR = os.path.join(os.getcwd(), 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
for dir in [DATA_DIR, MODELS_DIR]:
	if not os.path.exists(dir):
		os.mkdir(dir)
			

# What model to download and load
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
#MODEL_NAME = 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
#MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
#MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28' # Accurate but heavy
#MODEL_NAME = 'centernet_resnet50_v2_512x512_coco17_tpu-8'


# Download and extract model
MODEL_DATE = '20200711'
MODEL_NAME = 'centernet_resnet101_v1_fpn_512x512_coco17_tpu-8'
MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
if not os.path.exists(PATH_TO_CKPT):
	print('Downloading model. This may take a while... ', end='')
	urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
	tar_file = tarfile.open(PATH_TO_MODEL_TAR)
	tar_file.extractall(MODELS_DIR)
	tar_file.close()
	os.remove(PATH_TO_MODEL_TAR)
	print('Done')

# Download labels file
LABEL_FILENAME = 'mscoco_label_map.pbtxt'
LABELS_DOWNLOAD_BASE = \
		'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
if not os.path.exists(PATH_TO_LABELS):
	print('Downloading label file... ', end='')
	urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
	print('Done')	

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		print("gpus: ",gpus)
		for gpu in gpus:
			#tf.config.experimental.set_memory_growth(gpu, True)
			tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])	
	except RuntimeError as e:
		print(e)


#Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()	

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
																	use_display_name=True)


def load_image_into_numpy_array(image):
	#ar = image.get_dataVis
	#ar = image
	ar = image[:, :, 0:3]
	##(im_height, im_width, channels) = image.get_data().shape
	(im_height, im_width, channels) = image.shape
	return np.array(ar).reshape((im_height, im_width, 3)).astype(np.uint8)


def load_depth_into_numpy_array(depth):
	#ar = depth.get_data()
	ar = depth[:, :, 0:4]
	(im_height, im_width, channels) = depth.shape
	return np.array(ar).reshape((im_height, im_width, channels)).astype(np.float32)

@tf.function
def detect_fn(image):
	"""Detect objects in image."""

	image, shapes = detection_model.preprocess(image)
	prediction_dict = detection_model.predict(image, shapes)
	detections = detection_model.postprocess(prediction_dict, shapes)

	return detections, prediction_dict, tf.reshape(shapes, [-1])


width = 640  #704
height = 480 #416
confidence = 0.55

image_np_global = np.zeros([width, height, 3], dtype=np.uint8)
depth_np_global = np.zeros([width, height, 4], dtype=np.float)

# exit_signal = False
# new_data = False

# # Limit to a maximum of 40% the GPU memory usage taken by TF https://www.tensorflow.org/guide/using_gpu
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5

# # What model to download and load
# #MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
# #MODEL_NAME = 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
# #MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
# #MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
# #MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28' # Accurate but heavy

# # Path to frozen detection graph. This is the actual model that is used for the object detection.
# #PATH_TO_FROZEN_GRAPH = '/home/jlew/git/tf_models/research/object_detection/inference_graph7910/frozen_inference_graph.pb'
# #PATH_TO_FROZEN_GRAPH = '/home/jlew/git/tf_models/research/object_detection/jae_inference_graph5811/frozen_inference_graph.pb'
# #PATH_TO_FROZEN_GRAPH = '/home/jlew/git/tf_models/research/object_detection/jae_inference_graph6102/frozen_inference_graph.pb' # aspect ratio 1:1
# #PATH_TO_FROZEN_GRAPH = '/home/jlew/git/tf_models/research/object_detection/jae_inference_graph10193/frozen_inference_graph.pb' # aspect ratio 1:1
# PATH_TO_FROZEN_GRAPH = '/home/jlew/git/tf_models/research/object_detection/jae_inference_graph18318/frozen_inference_graph.pb' # aspect ratio 1:1
# # Load a (frozen) Tensorflow model into memory.
# #print("Loading model " + MODEL_NAME)
# detection_graph = tf.Graph()
# with detection_graph.as_default():
# 	od_graph_def = tf.GraphDef()
# 	with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
# 		serialized_graph = fid.read()
# 		od_graph_def.ParseFromString(serialized_graph)
# 		tf.import_graph_def(od_graph_def, name='')


# # List of the strings that is used to add correct label for each b
# PATH_TO_LABELS = os.path.join('/home/jlew/git/raccoon_dataset', 'modex_label_map.pbtxt')
# NUM_CLASSES = 1

# # Loading label map
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
# 																use_display_name=True)
# category_index = label_map_util.create_category_index(categories)

# Object Detection Class

class Object_Detector:

	def __init__(self):
		self.image_pub = rospy.Publisher("/OD_image",Image, queue_size=1)
		self.object_pub = rospy.Publisher("/objects",Detection2DArray, queue_size=1)
		self.box_pub = rospy.Publisher("/detected_box",DetectedBox,queue_size=1)

		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/zed1/right/image_rect_color_f", Image, self.rs_image_cb, queue_size=1)
		self.depth_sub = rospy.Subscriber('/zed1/point_cloud/cloud_registered', PointCloud2, self.rs_depth_cb, queue_size=1)
		self.sess = tf.Session(graph=detection_graph,config=config)
		#self.depth_np = np.zeros([width, height, 4], dtype=np.float)

	def rs_image_cb(self, data):

		# convert ros image to opencv image. copy needed in order to make array mutable.
		image_mat = np.copy(self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough"))
		#image_mat = np.copy(self.bridge.imgmsg_to_cv2(data, "bgr"))

		image_np_global = load_image_into_numpy_array(image_mat)
		image_np = np.copy(image_np_global)

		image_np_expanded = np.expand_dims(image_np, axis=0)
		input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
		detections, predictions_dict, shapes = detect_fn(input_tensor)

		label_id_offset = 1
		image_np_with_detections = image_np.copy()

		viz_utils.visualize_boxes_and_labels_on_image_array(
			image_np_with_detections,
			detections['detection_boxes'][0].numpy(),
			(detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
			detections['detection_scores'][0].numpy(),
			category_index,
			use_normalized_coordinates=True,
			max_boxes_to_draw=200,
			min_score_thresh=.30,
			agnostic_mode=False)

		# #image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

		# # Each box represents a part of the image where a particular object was detected.
		# boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
		# # Each score represent how level of confidence for each of the objects.
		# # Score is shown on the result image, together with the class label.
		# scores = detection_graph.get_tensor_by_name('detection_scores:0')
		# classes = detection_graph.get_tensor_by_name('detection_classes:0')
		# num_detections = detection_graph.get_tensor_by_name('num_detections:0')
		
		# # Actual detection.
		# (boxes, scores, classes, num_detections) = self.sess.run(
		# 				[boxes, scores, classes, num_detections],
		# 				feed_dict={image_tensor: image_np_expanded})

		# num_detections_ = num_detections.astype(int)[0]

		# # Visualization of the results of a detection.
		# image_np = self.display_objects_distances(
		# 				image_np,
		# 				self.depth_np,
		# 				num_detections_,
		# 				np.squeeze(boxes),
		# 				np.squeeze(classes).astype(np.int32),
		# 				np.squeeze(scores),
		# 				category_index)

		# cv2.imshow('ZED object detection', cv2.resize(image_np, (width*3, height*3)))
		image_msg = self.bridge.cv2_to_imgmsg(image_np,encoding="bgr8")
		self.image_pub.publish(image_msg)

		if cv2.waitKey(10) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
		else:
			sleep(0.01)
	
	
	def rs_depth_cb(self, data):

		self.depth_np = data
		return


	def read_depth(self,width, height, data):
		if (height >= data.height) or (width >= data.width):
			print("data: ",data.width, data.height)
			return -1

		data_output = pc2.read_points(data, field_names=("x","y","z"), skip_nans=False, uvs=[[width, height]])
		int_data = next(data_output)
		#rospy.loginfo("int_data" + str(int_data))
		return int_data
	


	def display_objects_distances(self, image_np, depth_jae, num_detections, boxes_, classes_, scores_, category_index):
		box_to_display_str_map = collections.defaultdict(list)
		box_to_color_map = collections.defaultdict(str)

		research_distance_box = 20

		for i in range(num_detections):
			if scores_[i] > confidence:
				box = tuple(boxes_[i].tolist())
				if classes_[i] in category_index.keys():
					class_name = category_index[classes_[i]]['name']
				display_str = str(class_name)
				# if not display_str:
				# 	display_str = '{}%'.format(int(100 * scores_[i]))
				# else:
				# 	display_str = '{}: {}%'.format(display_str, int(100 * scores_[i]))

				# Find object distance
				ymin, xmin, ymax, xmax = box
				#print xmin, xmax, ymin,ymax
				x_center = int(xmin * width + (xmax - xmin) * width * 0.5)
				y_center = int(ymin * height + (ymax - ymin) * height * 0.5)
				x_vect = []
				y_vect = []
				z_vect = []

				min_y_r = max(int(ymin * height), int(y_center - research_distance_box))
				min_x_r = max(int(xmin * width), int(x_center - research_distance_box))
				max_y_r = min(int(ymax * height), int(y_center + research_distance_box))
				max_x_r = min(int(xmax * width), int(x_center + research_distance_box))

				if min_y_r < 0: min_y_r = 0
				if min_x_r < 0: min_x_r = 0
				if max_y_r > height: max_y_r = height
				if max_x_r > width: max_x_r = width

				#print "x", min_x_r, max_x_r
				#print "y", min_y_r, max_y_r
				for j_ in range(min_y_r, max_y_r):
					for i_ in range(min_x_r, max_x_r):
				# 		#z = depth_np[j_, i_, 2]
						xyz = self.read_depth(i_, j_, depth_jae)
						if xyz != -1 and not np.isnan(xyz[1]):
								x_vect.append(xyz[1])
								y_vect.append(-xyz[0])
								z_vect.append(xyz[2])
								#print "xyz: ",xyz
							
				if len(x_vect) > 0:
				# 	#print "x_vect: ", x_vect
					x = np.median(x_vect)
				# 	#print "y_vect: ", y_vect
					y = np.median(y_vect)
					
					z = np.median(z_vect)
				# 	print "\n z =",z
				# 	print " zmax = ", np.max(z_vect)
				# 	print " zmin = ", np.min(z_vect)
				# 	#print " z_vect: ", z_vect
				# 	#distance = math.sqrt(x * x + y * y + z * z)
				# 	distance = z

					display_str = display_str + ": " + str('%2.1f, %2.1f, %2.1f' %(x,y,z)) + " m"

				#display_str = " "

				box_to_display_str_map[box].append(display_str)
				box_to_color_map[box] = vis_util.STANDARD_COLORS[classes_[i] % len(vis_util.STANDARD_COLORS)]
				h = (xmax - xmin)*width/x/56.
				w = (ymax - ymin)*height/x/56.

				self.box_pub.publish([x,y,z,w,h])

		for box, color in box_to_color_map.items():
			ymin, xmin, ymax, xmax = box

			vis_util.draw_bounding_box_on_image_array(
				image_np,
				ymin,
				xmin,
				ymax,
				xmax,
				color=color,
				thickness=4,
				display_str_list=box_to_display_str_map[box],
				use_normalized_coordinates=True)

		return image_np



def main(args):
	rospy.init_node('box_detector_node')
	
	obj=Object_Detector()
	
	#os.system('echo -ne "\033]0;OCCUPANCY PERCENTAGE\007"')
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("ShutDown")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
