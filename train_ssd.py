import argparse
import keras
from keras import backend as K
from keras.preprocessing import image
import tensorflow as tf
import horovod.keras as hvd
import os
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras.models import load_model
from math import ceil
import numpy as np

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

parser = argparse.ArgumentParser(description='Keras SSD Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=None,
                    help='path to training data')
parser.add_argument('--val-dir', default=None,
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.h5',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.001,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

args = parser.parse_args()

# Horovod: initialize Horovod.
hvd.init()

# ----------- ssd configure begin -----------
img_height = 300
img_width = 300
img_channels = 3
mean_color = [123, 117, 104]
swap_channels = [2, 1, 0]
n_classes = 20
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5], [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5], [1.0, 2.0, 0.5]]
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300]
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
clip_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = True
# -----------  ssd configure end  -----------


# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

model = ssd_300(
    image_size = (img_height, img_width, img_channels),
    n_classes = n_classes,
    mode = 'training',
    l2_regularization = 0.0005,
    scales = scales,
    aspect_ratios_per_layer = aspect_ratios,
    two_boxes_for_ar1 = two_boxes_for_ar1,
    steps = steps,
    offsets = offsets,
    clip_boxes = clip_boxes,
    variances = variances,
    normalize_coords = normalize_coords,
    subtract_mean = mean_color,
    swap_channels = swap_channels
)

weights_path = "./models/VGG_ILSVRC_16_layers_fc_reduced.h5"
model.load_weights(weights_path, by_name=True)

# : Instantiate an optimizer and the SSD loss function and compile the model.
#    If you want to follow the original Caffe implementation, use the preset SGD
#    optimizer, otherwise I'd recommend the commented-out Adam optimizer.

#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break

# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')

# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0


# SSD dataloader
train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

VOC_2007_images_dir = os.path.join(args.train_dir, "VOC2007/JPEGImages/")
VOC_2012_images_dir = os.path.join(args.train_dir, "VOC2012/JPEGImages/")

# The directories that contain the annotations.
VOC_2007_annotations_dir = os.path.join(args.train_dir, "VOC2007/Annotations/") 
VOC_2012_annotations_dir = os.path.join(args.train_dir, "VOC2012/Annotations/")

# The paths to the image sets.
VOC_2007_train_image_set_filename    = os.path.join(args.train_dir, "VOC2007/ImageSets/Main/train.txt")
VOC_2012_train_image_set_filename    = os.path.join(args.train_dir, "VOC2012/ImageSets/Main/train.txt")
VOC_2007_val_image_set_filename      = os.path.join(args.train_dir, "VOC2007/ImageSets/Main/val.txt")
VOC_2012_val_image_set_filename      = os.path.join(args.train_dir, "VOC2012/ImageSets/Main/val.txt")
VOC_2007_trainval_image_set_filename = os.path.join(args.train_dir, "VOC2007/ImageSets/Main/trainval.txt")
VOC_2012_trainval_image_set_filename = os.path.join(args.train_dir, "VOC2012/ImageSets/Main/trainval.txt")
VOC_2007_test_image_set_filename     = os.path.join(args.train_dir, "VOC2007/ImageSets/Main/test.txt")

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor'
]

train_dataset.parse_xml(
    images_dirs = [VOC_2007_images_dir, VOC_2012_images_dir],
    image_set_filenames = [VOC_2007_trainval_image_set_filename, VOC_2012_trainval_image_set_filename],
    annotations_dirs=[VOC_2007_annotations_dir, VOC_2012_annotations_dir],
    classes=classes,
    include_classes='all',
    exclude_truncated=False,
    exclude_difficult=False,
    ret=False
)

val_dataset.parse_xml(
    images_dirs=[VOC_2007_images_dir],
    image_set_filenames=[VOC_2007_test_image_set_filename],
    annotations_dirs=[VOC_2007_annotations_dir],
    classes=classes,
    include_classes='all',
    exclude_truncated=False,
    exclude_difficult=True,
    ret=False
)

train_dataset.create_hdf5_dataset(
    file_path='dataset_pascal_voc_07+12_trainval-%d.h5'% hvd.rank(),
    resize=False,
    variable_image_size=True,
    verbose=False
)

val_dataset.create_hdf5_dataset(
    file_path='dataset_pascal_voc_07_test-%d.h5'% hvd.rank(),
    resize=False,
    variable_image_size=True,
    verbose=False
)
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height, img_width=img_width, background=mean_color)


# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

predictor_sizes = [
    model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
    model.get_layer('fc7_mbox_conf').output_shape[1:3],
    model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
    model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
    model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
    model.get_layer('conv9_2_mbox_conf').output_shape[1:3]
]

ssd_input_encoder = SSDInputEncoder(
    img_height = img_height,
    img_width               = img_width,
    n_classes               = n_classes,
    predictor_sizes         = predictor_sizes,
    scales                  = scales,
    aspect_ratios_per_layer = aspect_ratios,
    two_boxes_for_ar1       = two_boxes_for_ar1,
    steps                   = steps,
    offsets                 = offsets,
    clip_boxes              = clip_boxes,
    variances               = variances,
    matching_type           = 'multi',
    pos_iou_threshold       = 0.5,
    neg_iou_limit           = 0.5,
    normalize_coords        = normalize_coords
)

train_generator = train_dataset.generate(
    batch_size             = args.batch_size,
    shuffle                = True,
    transformations        = [ssd_data_augmentation],
    label_encoder          = ssd_input_encoder,
    returns                = {'processed_images', 'encoded_labels'},
    keep_images_without_gt = False
)

val_generator = val_dataset.generate(
    batch_size             = args.batch_size,
    shuffle                = False,
    transformations        = [convert_to_3_channels, resize],
    label_encoder          = ssd_input_encoder,
    returns                = {'processed_images', 'encoded_labels'},
    keep_images_without_gt = False
)

train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()


# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: adjust learning rate based on number of GPUs.
initial_lr = args.base_lr * hvd.size()

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard, or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(
        warmup_epochs = args.warmup_epochs,
        initial_lr = initial_lr,
        verbose = verbose
    ),

    # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
    hvd.callbacks.LearningRateScheduleCallback(
        start_epoch = args.warmup_epochs,
        end_epoch = 80,
        multiplier = 1.,
        initial_lr = initial_lr
    ),
    hvd.callbacks.LearningRateScheduleCallback(
        start_epoch=80,
        end_epoch=100,
        multiplier=1e-1,
        initial_lr=initial_lr
    ),
    hvd.callbacks.LearningRateScheduleCallback(
        start_epoch=100,
        multiplier=1e-2,
        initial_lr=initial_lr
    ),
]

initial_epoch   = 0
final_epoch     = 120
steps_per_epoch = 1000


# Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint(args.checkpoint_format))
    callbacks.append(keras.callbacks.TensorBoard(args.log_dir))

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=final_epoch,
    callbacks=callbacks,
    validation_data=val_generator,
    validation_steps=ceil(val_dataset_size/args.batch_size),
    initial_epoch=initial_epoch
)

# Evaluate the model on the full data set.
score = hvd.allreduce(model.evaluate_generator(test_iter, len(test_iter), workers=4))
if verbose:
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
