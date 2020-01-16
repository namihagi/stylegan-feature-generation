import tensorflow as tf

from training.src_for_face_detector import AnchorGenerator
from training.src_for_face_detector import Detector
from training.src_for_face_detector.network_input_feature import FeatureExtractor


def FaceDetector(images, features):
    images = images.set_shape([None, None, None, 3])
    features = features.set_shape([None, 32, 32, 128])

    with tf.variable_scope('student'):
        tf.get_variable_scope().reuse_variables()

        feature_extractor = FeatureExtractor(is_training=True)
        anchor_generator = AnchorGenerator()
        detector = Detector(features, images, feature_extractor, anchor_generator)
        return detector

        # with tf.name_scope('student_prediction'):
        #     prediction = detector.get_predictions(
        #         score_threshold=0.3, iou_threshold=0.3, max_boxes=200
        #     )


def load(sess, saver, ckpt_dir):
    print(" [*] Reading checkpoint for face detector ...")

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        return True
    else:
        return False
