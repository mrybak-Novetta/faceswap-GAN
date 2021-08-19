import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import keras.backend as K
from detector.face_detector import MTCNNFaceDetector
import os
import ntpath
from tqdm import tqdm
import imageio

from preprocess import preprocess_video
import ntpath
video_path = '/parent_dir/videos/frisbee/3_LeftToRight_resized.avi'

global TOTAL_ITERS
TOTAL_ITERS = 34000

fd = MTCNNFaceDetector(sess=K.get_session(), model_path="./mtcnn_weights/")
reader = imageio.get_reader(video_path)
image_directory_parent = '/parent_dir/videos/frisbee'
suffix = "3_LeftToRight_resized"
# z=1
# for i, image in tqdm(enumerate(reader)):
#     output_detection_path = os.path.join(image_directory_parent, 'faces')
#     if not os.path.exists(output_detection_path):
#         os.mkdir(output_detection_path)
#     output_detection_path_rgb = os.path.join(output_detection_path, 'rgb')
#     if not os.path.exists(output_detection_path_rgb):
#         os.mkdir(output_detection_path_rgb)
#     output_detection_path_binary_mask = os.path.join(output_detection_path, 'binary_mask')
#     if not os.path.exists(output_detection_path_binary_mask):
#         os.mkdir(output_detection_path_binary_mask)
#     filename = str(i).zfill(5) + '.png'
preprocess_video(video_path, fd, 3, save_path=image_directory_parent, suffix=suffix)



