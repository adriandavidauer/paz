import os
import argparse
import cv2
import numpy as np
import logging
from collections import defaultdict
from datetime import datetime

from ava_dataset import AvaDataset
import paz.pipelines.detection as dt


def setup_logging(log_file_path=None, enable_logging=True):
    """Setup logging to file."""
    # Create logger
    logger = logging.getLogger('VVAD_Evaluation')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    if not enable_logging:
        # Add a null handler to prevent "No handler found" warnings
        logger.addHandler(logging.NullHandler())
        return logger
    
    if log_file_path is None:
        # Create logs directory if it doesn't exist
        logs_dir = 'logs'
        os.makedirs(logs_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(logs_dir, f"vvad_evaluation_{timestamp}.log")
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(detailed_formatter)
    
    logger.addHandler(file_handler)
    
    return logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate DetectVVAD on an AVA video with multiple faces')
    parser.add_argument('--root_dir', type=str,
                        default=os.path.expanduser('~/PAZ/ava-data'),
                        help='Root directory of AVA data (with videos/ and annotations/)')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to a specific AVA video file to process')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (default: <video_basename>_vvad.mp4)')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for matching predicted boxes to GT boxes')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path (default: logs/vvad_evaluation_TIMESTAMP.log)')
    parser.add_argument('--enable_logging', type=bool, default=True,
                        help='Enable logging (default: True)')
    return parser.parse_args()


def pick_first_annotated_video(dataset: AvaDataset):
    """Find first video with available annotations. 
    In case the user does not specify a video, this function will pick the first video with available annotations."""
    for file_name in dataset.file_names:
        video_name = os.path.splitext(file_name)[0]
        csv_path = os.path.join(dataset.csv_dir, f"{video_name}-activespeaker.csv")
        if not os.path.exists(csv_path):
            continue
        annots = dataset._load_annotation_csv(video_name)
        if annots != []:
            return os.path.join(dataset.video_dir, file_name)
    return None


def _map_ava_to_vvad_text(ava_numeric_label: int) -> str:
    """Map AVA label (0,1,2) to VVAD label for comparison.
    
    Maps:
    - 0 (NOT_SPEAKING) -> 'not-speaking'
    - 1 (SPEAKING_BUT_NOT_AUDIBLE) -> 'speaking'
    - 2 (SPEAKING_AUDIBLE) -> 'speaking'
    
    Both speaking states map to 'speaking' since VVAD only outputs 'speaking' or 'not-speaking'.
    """
    logger = logging.getLogger('VVAD_Evaluation')
    if ava_numeric_label == 0:
        return 'not-speaking'
    elif ava_numeric_label == 1:
        return 'speaking'
    elif ava_numeric_label == 2:
        return 'speaking'
    else:
        logger.error(f"Invalid AVA label: {ava_numeric_label}")
        return 'error'




def compute_iou(box_a, box_b):
    """Calculate IoU between two boxes in [x_min, y_min, x_max, y_max] format."""
    logger = logging.getLogger('VVAD_Evaluation')
    
    # Debug: print coordinate formats
    logger.debug(f"Computing IoU between boxes:")
    logger.debug(f"  Box A: {box_a} (type: {type(box_a)})")
    logger.debug(f"  Box B: {box_b} (type: {type(box_b)})")
    
    # Ensure boxes are in the correct format
    try:
        x_min_a, y_min_a, x_max_a, y_max_a = box_a
        x_min_b, y_min_b, x_max_b, y_max_b = box_b
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid box format - {e}")
        return 0.0
    
    # Check for valid coordinates
    if x_max_a <= x_min_a or y_max_a <= y_min_a:
        logger.warning(f"Invalid box A coordinates: max <= min")
        return 0.0
    if x_max_b <= x_min_b or y_max_b <= y_min_b:
        logger.warning(f"Invalid box B coordinates: max <= min")
        return 0.0
    
    intersection = _compute_intersection(box_a, box_b)
    area_a, area_b = _compute_box_areas(box_a, box_b)
    union = area_a + area_b - intersection
    
    iou = 0.0 if union == 0 else intersection / union
    logger.debug(f"IoU calculation: intersection={intersection:.1f}, area_a={area_a:.1f}, area_b={area_b:.1f}, union={union:.1f}, iou={iou:.3f}")
    return iou


def _compute_intersection(box_a, box_b):
    """Compute intersection area between two boxes."""
    x_min_a, y_min_a, x_max_a, y_max_a = box_a
    x_min_b, y_min_b, x_max_b, y_max_b = box_b
    
    inner_x_min = max(x_min_a, x_min_b)
    inner_y_min = max(y_min_a, y_min_b)
    inner_x_max = min(x_max_a, x_max_b)
    inner_y_max = min(y_max_a, y_max_b)
    
    inner_w = max(0, inner_x_max - inner_x_min)
    inner_h = max(0, inner_y_max - inner_y_min)
    return inner_w * inner_h


def _compute_box_areas(box_a, box_b):
    """Compute areas of two boxes."""
    x_min_a, y_min_a, x_max_a, y_max_a = box_a
    x_min_b, y_min_b, x_max_b, y_max_b = box_b
    
    area_a = (x_max_a - x_min_a) * (y_max_a - y_min_a)
    area_b = (x_max_b - x_min_b) * (y_max_b - y_min_b)
    return area_a, area_b


def match_boxes_to_gt(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Match predicted boxes to ground truth boxes using IoU."""
    logger = logging.getLogger('VVAD_Evaluation')
    
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        logger.info(f"No matching possible: pred_boxes={len(pred_boxes)}, gt_boxes={len(gt_boxes)}")
        return []
    
    logger.info(f"Computing IoU matrix for {len(pred_boxes)} predictions vs {len(gt_boxes)} GT boxes:")
    
    # Print detailed IoU information for all combinations
    for pred_idx, pred_box in enumerate(pred_boxes):
        pred_coords = pred_box.coordinates
        logger.debug(f"  Pred {pred_idx}: coords={pred_coords}")
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_coords, gt_box['bbox_pixel'])
            entity_id = gt_box['entity_id']
            gt_label = gt_box['gt_vvad_label']
            logger.debug(f"    vs GT {gt_idx} (entity {entity_id}, label {gt_label}): IoU={iou:.3f}")
    
    matches = []
    used_gt_indices = set()
    
    for pred_idx, pred_box in enumerate(pred_boxes):
        match = _find_best_match(pred_idx, pred_box, gt_boxes, used_gt_indices, iou_threshold)
        if match:
            matches.append(match)
            used_gt_indices.add(match[1])
            logger.info(f"  Match found: Pred {pred_idx} -> GT {match[1]} (IoU: {match[2]:.3f})")
        else:
            logger.info(f"  No match for Pred {pred_idx} (best IoU below threshold {iou_threshold})")
    
    return matches


def _find_best_match(pred_idx, pred_box, gt_boxes, used_gt_indices, iou_threshold):
    """Find best ground truth match for a predicted box."""
    logger = logging.getLogger('VVAD_Evaluation')
    
    pred_coords = pred_box.coordinates
    best_iou = 0
    best_gt_idx = None
    
    for gt_idx, gt_box in enumerate(gt_boxes):
        if gt_idx in used_gt_indices:
            continue
        
        iou = compute_iou(pred_coords, gt_box['bbox_pixel'])
        if iou > best_iou:
            best_iou = iou
            best_gt_idx = gt_idx
    
    # Only return match if IoU meets threshold
    if best_iou >= iou_threshold:
        return (pred_idx, best_gt_idx, best_iou)
    else:
        # Print best IoU even if below threshold for debugging
        if best_gt_idx is not None:
            entity_id = gt_boxes[best_gt_idx]['entity_id']
            gt_label = gt_boxes[best_gt_idx]['gt_vvad_label']
            logger.debug(f"Best IoU for Pred {pred_idx}: {best_iou:.3f} with GT {best_gt_idx} (entity {entity_id}, label {gt_label}) - BELOW threshold {iou_threshold}")
        return None


def initialize_video_capture(video_path: str):
    """Initialize video capture and get basic properties."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video file: {video_path}')
    return cap


def get_video_properties(cap, dataset: AvaDataset):
    """Extract video properties and validate first frame."""
    fps = cap.get(cv2.CAP_PROP_FPS) or dataset.target_fps
    if fps <= 0:
        fps = dataset.target_fps
    
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise ValueError('Could not read first frame')
    
    height, width = first_frame.shape[:2]
    if height <= 0 or width <= 0:
        cap.release()
        raise ValueError(f'Invalid frame dimensions: {width}x{height}')
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return fps, width, height


def get_evaluation_timestamps(annots, min_frames=38, stride=2):
    """Get timestamps where VVAD predictions should be evaluated."""
    logger = logging.getLogger('VVAD_Evaluation')
    
    if not annots:
        return []
    
    timestamps = sorted(set(annot['timestamp'] for annot in annots))
    if not timestamps:
        return []
    
    logger.info(f'All available timestamps: {len(timestamps)} total')
    logger.info(f'Timestamp range: {timestamps[0]:.2f} to {timestamps[-1]:.2f} seconds')
    logger.debug(f'Sample timestamps: {timestamps[:10]}')
    
    # Start from min_frames to allow VVAD buffer to mature
    start_idx = min(min_frames, len(timestamps))
    eval_timestamps = timestamps #timestamps[start_idx::stride]
    logger.info(f'After applying min_frames={min_frames} and stride={stride}: {len(eval_timestamps)} timestamps')
    logger.debug(f'Evaluation timestamps: {eval_timestamps[:10]}')
    
    return eval_timestamps


def timestamp_to_frame_index(timestamp, fps):
    """Convert timestamp in seconds to frame index."""
    return int(round(timestamp * fps))


def map_annotations_to_frames(annots, fps, dataset: AvaDataset):
    """Map annotations by timestamp to frame indices."""
    timestamp_to_annotations = {}
    
    for annot in annots:
        timestamp = annot.get('timestamp', 0)
        frame_idx = timestamp_to_frame_index(timestamp, fps)
        
        if frame_idx not in timestamp_to_annotations:
            timestamp_to_annotations[frame_idx] = []
        
        timestamp_to_annotations[frame_idx].append({
            'bbox_normalized': annot['bbox'],
            'label': annot['label'],
            'entity_id': annot.get('entity_id', ''),
            'timestamp': timestamp
        })
    
    return timestamp_to_annotations, 0.0, 1.0/fps


def convert_gt_boxes_to_pixel(gt_boxes_for_frame, width, height, dataset: AvaDataset):
    """Convert normalized GT boxes to pixel coordinates."""
    gt_boxes_pixel = []
    for gt_ann in gt_boxes_for_frame:
        gt_box_pixel = _convert_single_box_to_pixel(gt_ann, width, height, dataset)
        gt_boxes_pixel.append(gt_box_pixel)
    return gt_boxes_pixel


def _convert_single_box_to_pixel(gt_ann, width, height, dataset: AvaDataset):
    """Convert a single normalized box to pixel coordinates."""
    logger = logging.getLogger('VVAD_Evaluation')
    
    x_min_norm, y_min_norm, x_max_norm, y_max_norm = gt_ann['bbox_normalized']
    bbox_pixel = (
        x_min_norm * width, y_min_norm * height,
        x_max_norm * width, y_max_norm * height
    )
    
    # Debug: print coordinate conversion
    logger.debug(f"Converting GT box for entity {gt_ann['entity_id']}:")
    logger.debug(f"  Normalized: {gt_ann['bbox_normalized']}")
    logger.debug(f"  Pixel: {bbox_pixel}")
    logger.debug(f"  Image size: {width}x{height}")
    logger.debug(f"  Label: {gt_ann['label']} -> {_map_ava_to_vvad_text(dataset._map_label(gt_ann['label']))}")
    
    return {
        'bbox_pixel': bbox_pixel,
        'label': gt_ann['label'],
        'entity_id': gt_ann['entity_id'],
        'gt_vvad_label': _map_ava_to_vvad_text(dataset._map_label(gt_ann['label']))
    }


def process_frame_predictions(frame_rgb, pipeline):
    """Process frame through pipeline and get predictions."""
    logger = logging.getLogger('VVAD_Evaluation')
    
    result = pipeline(frame_rgb)
    out_rgb = result['image'] if isinstance(result, dict) and 'image' in result else frame_rgb
    pred_boxes = result.get('boxes2D', []) if isinstance(result, dict) else []
    
    # Debug: print prediction details and coordinate format
    logger.debug(f"Pipeline predictions:")
    logger.debug(f"Result raw predictions: {result}")
    logger.debug(f"  Result type: {type(result)}")
    logger.debug(f"  Number of predicted boxes: {len(pred_boxes)}")
    logger.debug(f"  Frame shape: {frame_rgb.shape}")
    
    for i, pred_box in enumerate(pred_boxes):
        coords = getattr(pred_box, 'coordinates', None)
        class_name = getattr(pred_box, 'class_name', 'None')
        score = getattr(pred_box, 'score', 'None')
        
        logger.debug(f"  Pred {i}: class_name={class_name}, score={score}")
        logger.debug(f"        coordinates={coords}")
        
        # Check if coordinates are normalized (0-1) or pixel coordinates
        if coords is not None and len(coords) >= 4:
            x_min, y_min, x_max, y_max = coords[:4]
            if 0 <= x_min <= 1 and 0 <= y_min <= 1 and 0 <= x_max <= 1 and 0 <= y_max <= 1:
                logger.debug(f"        → Coordinates appear to be NORMALIZED (0-1 range)")
            elif x_max > 1 or y_max > 1:
                logger.debug(f"        → Coordinates appear to be PIXEL coordinates (max: {x_max:.1f}, {y_max:.1f})")
            else:
                logger.debug(f"        → Coordinate format unclear")
    
    return out_rgb, pred_boxes


def draw_prediction_overlay(out_rgb, pred_box, gt_label, iou, width, height):
    """Draw prediction overlay on frame."""
    pred_label = getattr(pred_box, 'class_name', None)
    if pred_label is None:
        return
    
    status, color = _get_prediction_status(pred_label, gt_label)
    text_x, text_y = _get_text_position(pred_box, width, height)
    text = f"{pred_label} | {status} (IoU:{iou:.2%})"  # Show IoU as percentage
    
    try:
        cv2.putText(out_rgb, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    except Exception as e:
        logger = logging.getLogger('VVAD_Evaluation')
        logger.warning(f'Failed to draw text: {e}')


def _get_prediction_status(pred_label, gt_label):
    """Get prediction status and color based on correctness."""
    status = 'OK' if pred_label == gt_label else 'WRONG'
    color = (0, 255, 0) if pred_label == gt_label else (255, 0, 0)
    return status, color


def _get_text_position(pred_box, width, height):
    """Calculate text position for overlay."""
    x_min, y_min, x_max, y_max = pred_box.coordinates
    text_x = max(0, min(int(x_min), width - 200))
    text_y = max(20, min(int(y_min) - 10, height - 10))
    return text_x, text_y


def draw_frame_summary(out_rgb, frame_idx, frame_correct, frame_matched, height):
    """Draw frame-level accuracy summary."""
    try:
        if frame_matched > 0:
            frame_accuracy = frame_correct / frame_matched
            text = f"Frame {frame_idx}: {frame_correct}/{frame_matched} correct ({frame_accuracy:.2f})"
            color = (255, 255, 255)
        else:
            text = f"Frame {frame_idx}: No matches"
            color = (0, 255, 255)
        
        cv2.putText(out_rgb, text, (10, max(30, height - 30)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    except Exception as e:
        print(f'Warning: Failed to draw frame summary: {e}')


def initialize_entity_tracking():
    """Initialize entity-level tracking dictionaries."""
    return {
        'entity_predictions': defaultdict(int),
        'entity_correct': defaultdict(int),
        'entity_matched': defaultdict(int),
        'entity_labels': defaultdict(lambda: defaultdict(int)),
        'entity_gt_labels': defaultdict(lambda: defaultdict(int)),  # GT labels for matched annotations (mapped to VVAD)
        'entity_gt_total': defaultdict(int),  # Total GT annotations per entity from CSV
        'entity_gt_labels_all': defaultdict(lambda: defaultdict(int))  # All GT labels from CSV (original AVA labels)
    }


def update_entity_stats(entity_tracking, entity_id, pred_label, gt_label, is_correct):
    """Update entity-level statistics."""
    entity_tracking['entity_predictions'][entity_id] += 1
    entity_tracking['entity_labels'][entity_id][pred_label] += 1
    entity_tracking['entity_matched'][entity_id] += 1
    
    if is_correct:
        entity_tracking['entity_correct'][entity_id] += 1
    
    # Track ground truth labels for analysis
    if gt_label not in entity_tracking['entity_gt_labels'][entity_id]:
        entity_tracking['entity_gt_labels'][entity_id][gt_label] = 0
    entity_tracking['entity_gt_labels'][entity_id][gt_label] += 1


def print_entity_accuracy(entity_tracking):
    """Print entity-level accuracy statistics."""
    print('\nEntity-level Accuracy:')
    print('=' * 60)
    
    # Show all entities (both matched and unmatched)
    all_entity_ids = sorted(set(list(entity_tracking['entity_predictions'].keys()) + 
                               list(entity_tracking['entity_gt_total'].keys())))
    
    for entity_id in all_entity_ids:
        if entity_id in entity_tracking['entity_predictions']:
            _print_single_entity_stats(entity_id, entity_tracking)
        else:
            # Entity exists in GT but has no matches
            gt_total = entity_tracking['entity_gt_total'][entity_id]
            gt_labels_all = entity_tracking['entity_gt_labels_all'][entity_id]
            print(f'  Entity {entity_id}:')
            print(f'    No matches: 0/{gt_total} GT annotations matched')
            print(f'    Ground Truth (all from CSV): speaking-audible={gt_labels_all.get("speaking-audible", 0)}, '
                  f'speaking-not-audible={gt_labels_all.get("speaking-not-audible", 0)}, '
                  f'not-speaking={gt_labels_all.get("not-speaking", 0)}')
    
    _print_entity_summary(entity_tracking)


def _print_single_entity_stats(entity_id, entity_tracking):
    """Print statistics for a single entity."""
    total = entity_tracking['entity_predictions'][entity_id]
    correct = entity_tracking['entity_correct'][entity_id]
    matched = entity_tracking['entity_matched'][entity_id]
    accuracy = correct / matched if matched > 0 else 0
    
    pred_labels = entity_tracking['entity_labels'][entity_id]
    gt_labels = entity_tracking['entity_gt_labels'][entity_id]  # Matched GT labels (mapped to VVAD)
    gt_labels_all = entity_tracking['entity_gt_labels_all'][entity_id]  # All GT labels from CSV (original AVA)
    gt_total = entity_tracking['entity_gt_total'][entity_id]
    
    print(f'  Entity {entity_id}:')
    print(f'    Accuracy: {accuracy:.4f} ({correct}/{matched})')
    print(f'    Matched: {matched}/{gt_total} GT annotations')
    print(f'    Predictions: speaking={pred_labels.get("speaking", 0)}, '
          f'not-speaking={pred_labels.get("not-speaking", 0)}')
    print(f'    Ground Truth (matched): speaking={gt_labels.get("speaking", 0)}, '
          f'not-speaking={gt_labels.get("not-speaking", 0)}')
    print(f'    Ground Truth (all from CSV): speaking-audible={gt_labels_all.get("speaking-audible", 0)}, '
          f'speaking-not-audible={gt_labels_all.get("speaking-not-audible", 0)}, '
          f'not-speaking={gt_labels_all.get("not-speaking", 0)}')


def _print_entity_summary(entity_tracking):
    """Print summary statistics across all entities."""
    # Count all unique entities from ground truth (including unmatched)
    all_entity_ids = set(entity_tracking['entity_gt_total'].keys())
    matched_entity_ids = set(entity_tracking['entity_predictions'].keys())
    
    total_entities_gt = len(all_entity_ids)
    total_entities_matched = len(matched_entity_ids)
    
    total_correct = sum(entity_tracking['entity_correct'].values())
    total_matched = sum(entity_tracking['entity_matched'].values())
    total_gt_annotations = sum(entity_tracking['entity_gt_total'].values())
    overall_accuracy = total_correct / total_matched if total_matched > 0 else 0
    
    print('\nEntity Summary:')
    print(f'  Total unique entities in GT: {total_entities_gt}')
    print(f'  Entities with matches: {total_entities_matched}')
    print(f'  Entities without matches: {total_entities_gt - total_entities_matched}')
    print(f'  Total GT annotations: {total_gt_annotations}')
    print(f'  Total matched annotations: {total_matched}')
    print(f'  Overall entity accuracy: {overall_accuracy:.4f} ({total_correct}/{total_matched})')
    
    # Show unmatched entities
    unmatched_entities = all_entity_ids - matched_entity_ids
    if unmatched_entities:
        print(f'\n  Unmatched entities ({len(unmatched_entities)}):')
        for entity_id in sorted(unmatched_entities)[:10]:  # Show first 10
            gt_count = entity_tracking['entity_gt_total'][entity_id]
            print(f'    {entity_id}: {gt_count} GT annotations (no matches)')
        if len(unmatched_entities) > 10:
            print(f'    ... and {len(unmatched_entities) - 10} more')


def setup_evaluation(video_path: str, dataset: AvaDataset):
    """Setup evaluation components and return configuration."""
    pipeline = dt.DetectVVAD()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    annots = dataset._load_annotation_csv(video_name)
    
    if annots == []:
        raise ValueError(f'No annotations found for video {video_name}')
    
    return pipeline, video_name, annots


def setup_video_io(video_path: str, output_path: str, dataset: AvaDataset):
    """Initialize video capture and writer."""
    cap = initialize_video_capture(video_path)
    fps, width, height = get_video_properties(cap, dataset)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, dataset.target_fps, (width, height))
    
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Failed to open video writer for {output_path}')
    
    return cap, writer, fps, width, height


def initialize_ground_truth_entity_counts(annots, dataset: AvaDataset):
    """Initialize ground truth entity counts from all CSV annotations.
    
    Tracks original AVA labels separately from mapped VVAD labels to show
    the breakdown of speaking-and-audible vs speaking-but-not-audible.
    """
    entity_gt_total = defaultdict(int)
    entity_gt_labels_all = defaultdict(lambda: defaultdict(int))
    
    for annot in annots:
        entity_id = annot.get('entity_id', '')
        if entity_id:
            entity_gt_total[entity_id] += 1
            label = annot.get('label', '')
            if label:
                # Use label directly from CSV (convert to lowercase with dashes for consistency)
                label_key = label.lower().replace('_', '-')
                entity_gt_labels_all[entity_id][label_key] += 1
    
    return entity_gt_total, entity_gt_labels_all


def prepare_evaluation_data(annots, fps, dataset: AvaDataset):
    """Prepare evaluation timestamps and frame mappings."""
    logger = logging.getLogger('VVAD_Evaluation')
    
    eval_timestamps = get_evaluation_timestamps(annots)
    logger.info(f'Evaluating at {len(eval_timestamps)} timestamps')
    logger.debug(f'Evaluation timestamps: {eval_timestamps[:10]}...')  # Show first 10
    
    timestamp_to_annotations, _, _ = map_annotations_to_frames(annots, fps, dataset)
    
    logger.info(f'Total annotation frames: {len(timestamp_to_annotations)}')
    
    if len(timestamp_to_annotations) > 0:
        frame_range = f"{min(timestamp_to_annotations.keys())} to {max(timestamp_to_annotations.keys())}"
        logger.info(f'Frame indices range: {frame_range}')
        logger.debug(f'Sample annotated frames: {sorted(list(timestamp_to_annotations.keys()))[:10]}')
    
    eval_frame_indices = [timestamp_to_frame_index(ts, fps) for ts in eval_timestamps]
    logger.debug(f'Evaluation frame indices: {eval_frame_indices[:10]}...')  # Show first 10
    return timestamp_to_annotations, eval_frame_indices


def initialize_statistics():
    """Initialize statistics tracking dictionaries."""
    return {
        'total_predictions': 0,
        'total_matched': 0,
        'total_correct': 0,
        'speaking_count': 0,
        'not_speaking_count': 0,
        'entity_tracking': initialize_entity_tracking()
    }


def process_single_match(pred_box, gt_box, iou, width, height, stats, out_rgb):
    """Process a single matched prediction-ground truth pair."""
    pred_label = getattr(pred_box, 'class_name', None)
    if pred_label is None:
        return False
    
    stats['total_predictions'] += 1
    stats['speaking_count'] += pred_label == 'speaking'
    stats['not_speaking_count'] += pred_label == 'not-speaking'
    
    gt_label = gt_box['gt_vvad_label']
    is_correct = pred_label == gt_label
    
    if is_correct:
        stats['total_correct'] += 1
    
    entity_id = gt_box['entity_id']
    update_entity_stats(stats['entity_tracking'], entity_id, pred_label, gt_label, is_correct)
    draw_prediction_overlay(out_rgb, pred_box, gt_label, iou, width, height)
    
    return is_correct


def evaluate_frame(frame_count, pred_boxes, timestamp_to_annotations, width, height,
                  dataset: AvaDataset, iou_threshold, stats, out_rgb):
    """Evaluate a single frame against ground truth annotations."""
    logger = logging.getLogger('VVAD_Evaluation')
    
    gt_boxes_for_frame = timestamp_to_annotations.get(frame_count, [])
    gt_boxes_pixel = convert_gt_boxes_to_pixel(gt_boxes_for_frame, width, height, dataset)
    
    logger.info(f'=== Frame {frame_count} Evaluation ===')
    logger.info(f'Predicted boxes: {len(pred_boxes)}')
    logger.info(f'Ground truth boxes: {len(gt_boxes_for_frame)}')
    
    # Debug: Check if this frame should have annotations
    if frame_count not in timestamp_to_annotations:
        logger.warning(f'Frame {frame_count} not found in timestamp_to_annotations!')
        logger.debug(f'Available frames with annotations: {sorted(timestamp_to_annotations.keys())[:10]}...')  # Show first 10
    else:
        logger.debug(f'Frame {frame_count} found in annotations with {len(gt_boxes_for_frame)} entities')
    
    if len(gt_boxes_for_frame) > 0:
        logger.debug('Ground truth entities:')
        for i, gt_box in enumerate(gt_boxes_pixel):
            entity_id = gt_box['entity_id']
            gt_label = gt_box['gt_vvad_label']
            bbox = gt_box['bbox_pixel']
            logger.debug(f'  GT {i}: Entity {entity_id}, Label {gt_label}, Box {bbox}')
    else:
        logger.warning('No ground truth entities for this frame!')
        # Show nearby frames that might have annotations
        nearby_frames = [f for f in timestamp_to_annotations.keys() if abs(f - frame_count) <= 5]
        if nearby_frames:
            logger.debug(f'Nearby frames with annotations: {nearby_frames}')
    
    if len(pred_boxes) > 0:
        logger.debug('Predicted boxes:')
        for i, pred_box in enumerate(pred_boxes):
            pred_label = getattr(pred_box, 'class_name', 'Unknown')
            coords = pred_box.coordinates
            logger.debug(f'  Pred {i}: Label {pred_label}, Box {coords}')
    
    matches = match_boxes_to_gt(pred_boxes, gt_boxes_pixel, iou_threshold=iou_threshold)
    frame_correct = 0
    stats['total_matched'] += len(matches)
    
    logger.info(f'Matches found: {len(matches)}')
    
    # Show IoU for all entities that didn't get matched
    if len(gt_boxes_pixel) > 0 and len(pred_boxes) > 0:
        logger.info('IoU analysis for all entities:')
        for gt_idx, gt_box in enumerate(gt_boxes_pixel):
            entity_id = gt_box['entity_id']
            gt_label = gt_box['gt_vvad_label']
            best_iou = 0
            best_pred_idx = None
            
            for pred_idx, pred_box in enumerate(pred_boxes):
                iou = compute_iou(pred_box.coordinates, gt_box['bbox_pixel'])
                pred_label = getattr(pred_box, 'class_name', 'Unknown')
                logger.info(f'  Entity {entity_id} (GT {gt_idx}, {gt_label}) vs Pred {pred_idx} ({pred_label}): IoU = {iou:.2%}')
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            
            if best_iou < iou_threshold:
                logger.info(f'    → Best IoU for entity {entity_id}: {best_iou:.2%} (below threshold {iou_threshold:.2%})')
            else:
                logger.info(f'    → Best IoU for entity {entity_id}: {best_iou:.2%} (matched with Pred {best_pred_idx})')
    
    for pred_idx, gt_idx, iou in matches:
        pred_box = pred_boxes[pred_idx]
        gt_box = gt_boxes_pixel[gt_idx]
        entity_id = gt_box['entity_id']
        gt_label = gt_box['gt_vvad_label']
        pred_label = getattr(pred_box, 'class_name', 'Unknown')
        
        logger.info(f'Processing match: Pred {pred_idx} ({pred_label}) -> GT {gt_idx} (entity {entity_id}, {gt_label}) IoU: {iou:.2%}')
        
        if process_single_match(pred_box, gt_box, iou, width, height, stats, out_rgb):
            frame_correct += 1
            logger.info(f'  ✓ Correct prediction')
        else:
            logger.info(f'  ✗ Wrong prediction')
    
    draw_frame_summary(out_rgb, frame_count, frame_correct, len(matches), height)
    logger.info(f'Frame {frame_count} summary: {frame_correct}/{len(matches)} correct ({len(matches)} matches total)')
    return frame_correct


def process_video_frames(cap, writer, pipeline, timestamp_to_annotations, eval_frame_indices,
                        width, height, dataset: AvaDataset, iou_threshold, stats):
    """Process all video frames and perform evaluation."""
    frame_count = 0
    max_frame = max(eval_frame_indices) if eval_frame_indices else 0
    
    while frame_count <= max_frame:
        frame_bgr = _read_frame_safe(cap, frame_count)
        if frame_bgr is None:
            break
        
        frame_count = _process_single_frame(frame_bgr, frame_count, writer, pipeline,
                                           timestamp_to_annotations, eval_frame_indices,
                                           width, height, dataset, iou_threshold, stats)
    
    return frame_count


def _read_frame_safe(cap, frame_count):
    """Safely read a frame from video capture."""
    ret, frame_bgr = cap.read()
    if not ret or frame_bgr is None:
        return None
    return frame_bgr


def _process_single_frame(frame_bgr, frame_count, writer, pipeline, timestamp_to_annotations,
                         eval_frame_indices, width, height, dataset, iou_threshold, stats):
    """Process a single video frame."""
    try:
        if frame_bgr.size == 0:
            return frame_count + 1
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out_rgb, pred_boxes = process_frame_predictions(frame_rgb, pipeline)
        
        if frame_count in eval_frame_indices:
            evaluate_frame(frame_count, pred_boxes, timestamp_to_annotations,
                         width, height, dataset, iou_threshold, stats, out_rgb)
        
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        writer.write(out_bgr)
        return frame_count + 1
        
    except Exception as e:
        logger = logging.getLogger('VVAD_Evaluation')
        logger.error(f'Error processing frame {frame_count}: {e}')
        return frame_count + 1


def print_final_statistics(video_path: str, output_path: str, frame_count, fps,
                          eval_frame_indices, stats):
    """Print final evaluation statistics."""
    accuracy = _compute_overall_accuracy(stats)
    
    print('\nProcessing complete:')
    _print_basic_stats(video_path, output_path, frame_count, fps, eval_frame_indices, stats)
    _print_accuracy_stats(accuracy, stats)
    print_entity_accuracy(stats['entity_tracking'])


def _compute_overall_accuracy(stats):
    """Compute overall accuracy from statistics."""
    return (stats['total_correct'] / stats['total_matched']) if stats['total_matched'] > 0 else None


def _print_basic_stats(video_path, output_path, frame_count, fps, eval_frame_indices, stats):
    """Print basic processing statistics."""
    print(f'  Input:   {video_path}')
    print(f'  Output:  {output_path}')
    print(f'  Total frames processed: {frame_count} @ {fps:.2f} FPS')
    print(f'  Evaluation frames: {len(eval_frame_indices)}')
    print(f'  Total predictions: {stats["total_predictions"]}')
    print(f'  Total matched boxes: {stats["total_matched"]}')
    print(f"  speaking: {stats['speaking_count']}, not-speaking: {stats['not_speaking_count']}")


def _print_accuracy_stats(accuracy, stats):
    """Print accuracy-related statistics."""
    if accuracy is None:
        print('  Accuracy: N/A (no matched boxes)')
    else:
        print(f'  Overall Accuracy: {accuracy:.4f} ({stats["total_correct"]}/{stats["total_matched"]})')


def evaluate_ava_video(video_path: str, output_path: str, dataset: AvaDataset, iou_threshold: float):
    """Evaluate DetectVVAD on an AVA video with multiple faces."""
    pipeline, video_name, annots = setup_evaluation(video_path, dataset)
    cap, writer, fps, width, height = setup_video_io(video_path, output_path, dataset)
    
    timestamp_to_annotations, eval_frame_indices = prepare_evaluation_data(annots, fps, dataset)
    stats = initialize_statistics()
    
    # Initialize ground truth entity counts from all annotations
    entity_gt_total, entity_gt_labels_all = initialize_ground_truth_entity_counts(annots, dataset)
    stats['entity_tracking']['entity_gt_total'] = entity_gt_total
    stats['entity_tracking']['entity_gt_labels_all'] = entity_gt_labels_all
    
    try:
        frame_count = process_video_frames(cap, writer, pipeline, timestamp_to_annotations,
                                         eval_frame_indices, width, height, dataset,
                                         iou_threshold, stats)
    finally:
        cap.release()
        try:
            writer.release()
        except Exception as e:
            logger = logging.getLogger('VVAD_Evaluation')
            logger.warning(f'Error releasing video writer: {e}')
    
    print_final_statistics(video_path, output_path, frame_count, fps, eval_frame_indices, stats)


def main():
    """Main function to run VVAD evaluation on AVA video."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file, args.enable_logging)
    if args.enable_logging:
        logger.info('Starting VVAD evaluation on AVA video')
        logger.info(f'Arguments: {vars(args)}')
    
    dataset = AvaDataset(root_dir=args.root_dir)
    
    if args.video is None:
        video_path = pick_first_annotated_video(dataset)
        if video_path is None:
            raise RuntimeError('No annotated AVA videos found')
        if args.enable_logging:
            logger.info(f'Using first annotated video: {video_path}')
    else:
        video_path = args.video
        if not os.path.isabs(video_path):
            video_path = os.path.join(dataset.video_dir, video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f'Video not found: {video_path}')
        if args.enable_logging:
            logger.info(f'Using specified video: {video_path}')

    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(os.path.basename(video_path))
        output_path = os.path.join(os.path.dirname(video_path), f"{base}_vvad.mp4")
    
    if args.enable_logging:
        logger.info(f'Output will be saved to: {output_path}')

    evaluate_ava_video(
        video_path=video_path,
        output_path=output_path,
        dataset=dataset,
        iou_threshold=args.iou_threshold
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        exit(1)
