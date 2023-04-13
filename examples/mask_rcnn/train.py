import os
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model

from mask_rcnn.pipeline import DetectionPipeline
from paz.models.detection.utils import create_prior_boxes
from mask_rcnn.utils2 import DataGenerator

from mask_rcnn.shapes_loader import Shapes
from mask_rcnn.model import MaskRCNN, get_imagenet_weights
import numpy as np
import cv2

from tensorflow.python.keras.layers import Layer, Input, Lambda
from mask_rcnn.layers import DetectionTargetLayer, ProposalLayer
from mask_rcnn.loss_end_point import ProposalBBoxLoss, ProposalClassLoss,\
    BBoxLoss, ClassLoss, MaskLoss


# Extra arguments to be passed to model from default values
description = 'Training script for Mask RCNN model'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-bs', '--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('-dp', '--data_path', default='',
                    required=False, type=str, help='Directory for loading data')
parser.add_argument('-sp', '--save_path', default='',
                    required=False, metavar='/path/to/save',
                    help="Path to save model weights and logs")
parser.add_argument('-lr', '--learning_rate', default=0.002, type=float,
                    help='Initial learning rate for SGD')
parser.add_argument('-st', '--steps_per_epoch', default=1000, type=int,
                    help='steps per epoch for training')
parser.add_argument('-m', '--momentum', default=0.9, type=float,
                    help='Momentum for SGD')
parser.add_argument('-e', '--num_epochs', default=5, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('-et', '--evaluation_period', default=1, type=int,
                    help='evaluation frequency')
parser.add_argument('--AP_IOU', default=0.5, type=float,
                    help='Average precision IOU used for evaluation')
parser.add_argument('-l', '--layers', default='heads', type=str,
                    help='Select which layers to train')
parser.add_argument('-w', '--workers', default=1, type=int,
                    help='Number of workers used for optimization')
parser.add_argument('-mp', '--multiprocessing', default=False, type=bool,
                    help='Select True for multiprocessing')
parser.add_argument('-i', '--image_shape', default=np.array([128, 128, 3]), type=int,
                    help='Input image size')
parser.add_argument('-igpu', '--images_per_gpu', default=8, type=int,
                    help='Select no. of images to train per gpu')
parser.add_argument('-as', '--anchor_scales', default=(8, 16, 32, 64, 128), type=int,
                    help='Length of square anchor side in pixels')
parser.add_argument('-rois', '--rois_per_image', default=32, type=int,
                    help='Number of ROIs per image to feed to classifier/mask heads')
parser.add_argument('-nc', '--num_classes', default=4, type=int,
                    help='Number of classes')
parser.add_argument('-vs', '--valid_steps', default=5, type=int,
                    help='Number of validation steps')
parser.add_argument('-ai', '--anchors_per_image', default=256, type=int,
                    help='Number of anchors per image')

args = parser.parse_args()
print('Path to save model: ', args.save_path)
print('Data path: ', args.data_path)

# Dataset initialisation

optimizer = SGD(args.learning_rate, args.momentum, clipnorm=5.0)

dataset_train = Shapes(500, (args.image_shape[0], args.image_shape[1]))
dataset_val = Shapes(50, (args.image_shape[0], args.image_shape[1]))

train_generator = DataGenerator(dataset_train, "resnet101", args.image_shape, args.anchor_scales,
                                args.batch_size, args.num_classes, shuffle=True)
val_generator = DataGenerator(dataset_val, "resnet101", args.image_shape, args.anchor_scales,
                              args.batch_size, args.num_classes, shuffle=True)

# Initial model description
model = MaskRCNN(model_dir=args.data_path, image_shape=args.image_shape, backbone="resnet101",
                 batch_size=args.batch_size, images_per_gpu=args.images_per_gpu,
                 rpn_anchor_scales=args.anchor_scales, train_rois_per_image=args.rois_per_image,
                 num_classes=args.num_classes)

# Network head creation
model.build_train_model()

model.keras_model.load_weights('weights/mask_rcnn_coco.h5', by_name=True, skip_mismatch=True)

# Set which layers to train in the backbone Default: Heads
layer_regex = {
    # all layers but the backbone
    'heads': r'(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)',
    # From a specific Resnet stage and up
    '3+': r'(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)',
    '4+': r'(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)',
    '5+': r'(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)',
    # All layers
    'all': '.*',
}

layers = []
if args.layers in layer_regex.keys():
    layers = layer_regex[args.layers]
model.set_trainable(layer_regex=layers)

# Add losses and compile
rpn_class_loss = ProposalClassLoss()
rpn_bbox_loss = ProposalBBoxLoss(args.anchors_per_image, args.images_per_gpu)

custom_losses = {
        "rpn_class_logits": rpn_class_loss,
        "rpn_bbox": rpn_bbox_loss
        }

loss_weights = {
        'rpn_class_logits': 1.,
        'rpn_bbox': 1.,
    }


loss_names = ['mrcnn_class_loss', 'mrcnn_bbox_loss', 'mrcnn_mask_loss']

added_loss_name = []
for name in loss_names:
    layer = model.keras_model.get_layer(name)
    if layer.output in model.keras_model.losses:
        continue
    loss = (tf.reduce_mean(input_tensor=layer.output, keepdims=True))
    model.keras_model.add_loss(loss)
    added_loss_name.append(layer.name)


reg_losses = [
    l2(0.0001)(w) / tf.cast(tf.size(w), tf.float32)
    for w in model.keras_model.trainable_weights
    if 'gamma' not in w.name and 'beta' not in w.name]

model.keras_model.add_loss(tf.add_n(reg_losses))

model.keras_model.compile(optimizer=optimizer, loss=custom_losses, loss_weights=loss_weights)

for name in loss_names:
    if name in model.keras_model.metrics_names:
        continue
    layer = model.keras_model.get_layer(name)
    loss = (tf.math.reduce_mean(input_tensor=layer.output, keepdims=True))
    model.keras_model.add_metric(loss, name=name, aggregation='mean')

# Checkpoints
model_path = os.path.join(args.save_path, 'shapes')
if not os.path.exists(model_path):
    os.makedirs(model_path)

log = CSVLogger(os.path.join(model_path, 'shapes' + '-optimization.log'))
save_path = os.path.join(model_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
checkpoint = ModelCheckpoint(save_path, verbose=1, save_weights_only=True)
early_stop = EarlyStopping(monitor='loss', patience=3)

model.keras_model.fit(
    train_generator,
    epochs=args.num_epochs,
    steps_per_epoch=args.steps_per_epoch,
    callbacks=[log, checkpoint, early_stop],
    validation_data=val_generator,
    validation_steps=args.valid_steps,
    max_queue_size=100,
    workers=args.workers,
    use_multiprocessing=args.multiprocessing,
)
