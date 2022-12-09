import numpy as np
from paz.backend.boxes import to_center_form


def compute_feature_sizes(image_size, max_level):
    """Computes EfficientNet input feature size.

    # Arguments
        image_size: Tuple, (Height, Width) of input image.
        max_level: Int, maximum features levels.

    # Returns
        feature_sizes: Numpy array of shape ``(8, 2)``.
    """
    feature_H, feature_W = image_size
    feature_sizes = np.array([feature_H, feature_W], dtype=np.float64)
    for _ in range(1, max_level + 1):
        feature_H = (feature_H - 1) // 2 + 1
        feature_W = (feature_W - 1) // 2 + 1
        feature_size = np.array([feature_H, feature_W])
        feature_sizes = np.vstack((feature_sizes, feature_size))
    return feature_sizes


def generate_configurations(feature_sizes, min_level, max_level,
                            num_scales, aspect_ratios, anchor_scale):
    """Generates anchor box parameter combinations.

    # Arguments:
        feature_sizes: Numpy array, being input feature sizes.
        min_level: Int, being first EfficientNet layer index.
        max_level: Int, being last EfficientNet layer index.
        num_scales: Int, being number of anchor box scales.
        aspect_ratios: List, being anchor boxes aspect ratios.
        anchor_scale: Numpy array, being anchor box scales.

    # Returns:
        Tuple: being generated configuarations.
    """
    num_levels = max_level + 1 - min_level
    scale_aspect_ratio_combinations = (
        len(range(num_scales)) * len(aspect_ratios))
    features_H, features_W = build_features(
        feature_sizes, min_level, max_level, scale_aspect_ratio_combinations)
    octave_scales = build_octaves(num_scales, aspect_ratios, num_levels)
    aspects = build_aspects(aspect_ratios, num_scales, num_levels)
    strides_y, strides_x = build_strides(
        feature_sizes, features_H, features_W, num_levels)
    anchor_scales = build_scales(
        anchor_scale, scale_aspect_ratio_combinations, num_levels)
    return ((strides_y, strides_x, octave_scales, aspects, anchor_scales),
            num_levels, scale_aspect_ratio_combinations)


def build_strides(feature_sizes, features_H, features_W, num_levels):
    """Generates layer-wise EfficientNet anchor box strides.

    # Arguments:
        feature_sizes: Numpy array being input feature sizes.
        features_H: Numpy array being input feature height.
        features_W: Numpy array being input feature width.
        num_levels: Int, representing number of feature levels.

    # Returns:
        Tuple: Containing strides in y and x direction.
    """
    base_feature_H, base_feature_W = feature_sizes[0]
    strides_y = np.reshape(
        base_feature_H*np.reciprocal(features_H), (num_levels, -1))
    strides_x = np.reshape(
        base_feature_W*np.reciprocal(features_W), (num_levels, -1))
    return strides_y, strides_x


def build_features(feature_sizes, min_level, max_level,
                   scale_aspect_ratio_combinations):
    """Calculates layer-wise EfficientNet feature height and width.

    # Arguments:
        feature_sizes: Numpy array being input feature sizes.
        min_level: Int, being first EfficientNet layer index.
        max_level: Int, being last EfficientNet layer index.
        scale_aspect_ratio_combinations: Int, number of
            scales aspect ratios combinations.

    # Returns:
        Tuple: Containing feature height and width.
    """
    feature_H = feature_sizes[min_level: max_level + 1][:, 0]
    feature_W = feature_sizes[min_level: max_level + 1][:, 1]
    features_H = np.repeat(feature_H, scale_aspect_ratio_combinations)
    features_W = np.repeat(feature_W, scale_aspect_ratio_combinations)
    return features_H, features_W


def build_octaves(num_scales, aspect_ratios, num_levels):
    """Generates layer-wise EfficientNet anchor box octaves.

    # Arguments:
        num_scales: Int, being number of anchor box scales.
        aspect_ratios: List, being anchor boxes aspect ratios.
        num_levels: Int, being number of feature levels.

    # Returns:
        octave_scales: Numpy array of shape ``(5, 9)``.
    """
    scale_octaves = np.repeat(list(range(num_scales)), len(aspect_ratios))
    octave_scales = (np.tile(scale_octaves, num_levels)
                     / float(num_scales)).reshape(num_levels, -1)
    return octave_scales


def build_aspects(aspect_ratios, num_scales, num_levels):
    """Generates layer-wise EfficientNet anchor box aspect ratios.

    # Arguments:
        aspect_ratios: List, being anchor boxes aspect ratios.
        num_scales: Int, being number of anchor box scales.
        num_levels: Int, being number of feature levels.

    # Returns:
        aspects: Numpy array of shape ``(5, 9)``.
    """
    aspect = np.tile(aspect_ratios, len(range(num_scales)))
    aspects = np.tile(aspect, num_levels).reshape(num_levels, -1)
    return aspects


def build_scales(anchor_scale, scale_aspect_ratio_combinations, num_levels):
    """Generates layer-wise EfficientNet anchor box scales.

    # Arguments:
        anchor_scale: Numpy array, being anchor box scales.
        scale_aspect_ratio_combinations:  Int, number of
            scale aspect ratio combinations.
        num_levels: Int, being number of feature levels.

    # Returns:
        anchor_scales: Numpy array of shape ``(5, 9)``.
    """
    anchor_scales = np.reshape(
        np.repeat(anchor_scale, scale_aspect_ratio_combinations),
        (num_levels, -1))
    return anchor_scales


def compute_aspect_ratio(aspect):
    """Calculates anchor box aspect ratio.

    # Arguments:
        aspect: Numpy array being anchor box aspects.

    # Returns:
        Tuple: being anchor box aspect ratios.
    """
    aspect_x = np.sqrt(aspect)
    aspect_y = 1 / aspect_x
    return aspect_x, aspect_y


def compute_box_coordinates(stride_y, stride_x, octave_scale, aspect,
                            anchor_scale, image_size):
    """Calculates anchor box coordinates in centre form.

    # Arguments:
        stride_y: Numpy array, being y-direction stride.
        stride_x: Numpy array, being x-direction stride.
        octave_scale: Numpy array, being anchor box octave scale.
        aspect: Numpy array being anchor box aspects.
        anchor_scale: Numpy array, being anchor box scales.
        image_size: Tuple, being input image size.

    # Returns:
        Tuple: holding anchor box centre, width and height.
    """
    W, H = image_size
    base_anchor_x = anchor_scale * stride_x * (2 ** octave_scale)
    base_anchor_y = anchor_scale * stride_y * (2 ** octave_scale)
    aspect_x, aspect_y = compute_aspect_ratio(aspect)
    anchor_x = (base_anchor_x * aspect_x / 2.0) / H
    anchor_y = (base_anchor_y * aspect_y / 2.0) / W
    x = np.arange(stride_x / 2, H, stride_x)
    y = np.arange(stride_y / 2, W, stride_y)
    center_x, center_y = np.meshgrid(x, y)
    center_x = center_x.reshape(-1) / H
    center_y = center_y.reshape(-1) / W
    return center_x, center_y, anchor_x, anchor_y


def generate_level_boxes(strides_y, strides_x, octave_scales, aspects,
                         anchor_scales, image_size,
                         scale_aspect_ratio_combinations):
    """Generates anchor box in centre form per feature level.

    # Arguments:
        strides_y: Numpy array, being y-direction strides.
        strides_x: Numpy array, being x-direction strides.
        octave_scales: Numpy array, being anchor box octave scales.
        aspects: Numpy array being anchor box aspects.
        anchor_scales: Numpy array being anchor box scales.
        image_size: Tuple, being input image size.
        scale_aspect_ratio_combinations: Int, number of
            scale aspect ratio combinations.

    # Returns:
        boxes_level: List being anchor boxes in centre form.
    """
    boxes_level = []
    for combination in range(scale_aspect_ratio_combinations):
        box_coordinates = compute_box_coordinates(
            strides_y[combination], strides_x[combination],
            octave_scales[combination], aspects[combination],
            anchor_scales[combination], image_size)
        center_x, center_y, anchor_x, anchor_y = box_coordinates
        boxes = np.vstack((center_x - anchor_x, center_y - anchor_y,
                           center_x + anchor_x, center_y + anchor_y))
        boxes = np.swapaxes(boxes, 0, 1)
        boxes_level.append(np.expand_dims(boxes, axis=1))
    return boxes_level


def generate_anchors(feature_sizes, min_level, max_level, num_scales,
                     aspect_ratios, image_size, anchor_scales):
    """Generates anchor boxes in centre form for all feature levels.

    # Arguments:
        feature_sizes: Numpy array being input feature sizes.
        min_level: Int, being first EfficientNet layer index.
        max_level: Int, being last EfficientNet layer index.
        num_scales: Int, being number of anchor box scales.
        aspect_ratios: List, specifying anchor boxes aspect ratios.
        image_size: Tuple, representing input image size.
        anchor_scales: Numpy array being anchor box scales.

    # Returns:
        anchors: Numpy array of shape ``(49104, 4)``.
    """
    configuration = generate_configurations(
        feature_sizes, min_level, max_level, num_scales,
        aspect_ratios, anchor_scales)
    ((strides_y, strides_x, octave_scales, aspects, anchor_scales),
        num_levels, scale_aspect_ratio_combinations) = configuration
    boxes_all = []
    for level in range(num_levels):
        boxes_level = generate_level_boxes(
            strides_y[level], strides_x[level], octave_scales[level],
            aspects[level], anchor_scales[level], image_size,
            scale_aspect_ratio_combinations)
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))
    anchors = np.vstack(boxes_all).astype('float32')
    return anchors


def build_prior_boxes(min_level, max_level, num_scales,
                      aspect_ratios, anchor_scale, image_size):
    """Generates prior boxes.

    # Arguments
        min_level: Int, being first EfficientNet layer index.
        max_level: Int, being last EfficientNet layer index.
        num_scales: Int, being number of anchor box scales.
        aspect_ratios: List, specifying anchor boxes aspect ratios.
        anchor_scale: Float, being anchor box scale.
        image_size: Tuple, representing input image size.

    # Returns
        prior_boxes: Numpy array of shape ``(49104, 4)``.
    """
    anchor_scales = np.repeat(anchor_scale, max_level - min_level + 1)
    feature_sizes = compute_feature_sizes(image_size, max_level)
    prior_boxes = generate_anchors(
        feature_sizes, min_level, max_level, num_scales, aspect_ratios,
        image_size, anchor_scales)
    prior_boxes = to_center_form(prior_boxes)
    return prior_boxes
