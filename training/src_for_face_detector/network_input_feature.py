import tensorflow as tf
import tensorflow.contrib.slim as slim

from training.src_for_face_detector.constants import BATCH_NORM_MOMENTUM


class FeatureExtractor:
    def __init__(self, is_training):
        self.is_training = is_training
        self.features = None

    def __call__(self, input_features):
        """
        Arguments:
            images: a float tensor with shape [batch_size, height, width, 3],
                a batch of RGB images with pixel values in the range [0, 1].
        Returns:
            a list of float tensors where the ith tensor
            has shape [batch, height_i, width_i, channels_i].
        """

        def batch_norm(x):
            x = tf.layers.batch_normalization(
                x, axis=3, center=True, scale=True,
                momentum=BATCH_NORM_MOMENTUM, epsilon=0.001,
                training=self.is_training, fused=True,
                name='batch_norm'
            )
            return x

        # multiple scale convolutional layers
        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': 'NHWC'
        }
        with slim.arg_scope([slim.conv2d], **params):
            features = []  # extracted feature maps
            x = input_features
            self.features = x
            features.append(x)  # scale 0
            x = slim.conv2d(x, 128, (1, 1), scope='conv3_1')
            x = slim.conv2d(x, 256, (3, 3), stride=2, scope='conv3_2')
            features.append(x)  # scale 1
            x = slim.conv2d(x, 128, (1, 1), scope='conv4_1')
            x = slim.conv2d(x, 256, (3, 3), stride=2, scope='conv4_2')
            features.append(x)  # scale 2

        return features

    def get_feature_maps(self):
        return self.features


def preprocess(images):
    """Transform images before feeding them to the network."""
    return (2.0 * images) - 1.0


def inception_module(x, scope):
    # path 1
    x1 = slim.conv2d(x, 32, (1, 1), scope=scope + '/conv_1x1_path1')
    # path 2
    y = slim.max_pool2d(x, (3, 3), stride=1, padding='SAME', scope=scope + '/pool_3x3_path2')
    x2 = slim.conv2d(y, 32, (1, 1), scope=scope + '/conv_1x1_path2')
    # path 3
    y = slim.conv2d(x, 24, (1, 1), scope=scope + '/conv_1x1_path3')
    x3 = slim.conv2d(y, 32, (3, 3), scope=scope + '/conv_3x3_path3')
    # path 4
    y = slim.conv2d(x, 24, (1, 1), scope=scope + '/conv_1x1_path4')
    y = slim.conv2d(y, 32, (3, 3), scope=scope + '/conv_3x3_path4')
    x4 = slim.conv2d(y, 32, (3, 3), scope=scope + '/conv_3x3_second_path4')
    return tf.concat([x1, x2, x3, x4], axis=3, name=scope + '/concat')
