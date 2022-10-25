from paz.backend.image import show_image, write_image
from paz.datasets import get_class_names
from tensorflow.keras.utils import get_file

from detection import DetectSingleShot
from efficientdet import EFFICIENTDETD0
from utils import raw_images

WEIGHT_PATH = (
    'https://github.com/oarriaga/altamira-data/releases/download/v0.16/')
WEIGHT_FILE = 'efficientdet-d0-VOC-VOC_weights.hdf5'

if __name__ == "__main__":
    model = EFFICIENTDETD0(num_classes=21, base_weights='COCO',
                           head_weights=None)
    weights_path = get_file(WEIGHT_FILE, WEIGHT_PATH + WEIGHT_FILE,
                            cache_subdir='paz/models')
    model.load_weights(weights_path)
    detections = DetectSingleShot(model, get_class_names('VOC'),
                                  0.5, 0.45)(raw_images)
    show_image(detections['image'])
    write_image('predictions.png', detections['image'])
    print('task completed')
