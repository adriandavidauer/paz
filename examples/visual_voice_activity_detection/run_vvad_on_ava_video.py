import os
import argparse
import cv2
import numpy as np

from ava_dataset import AvaDataset
import paz.pipelines.detection as dt


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run DetectVVAD on an AVA video and save annotated output')
    parser.add_argument('--root_dir', type=str,
                        default=os.path.expanduser('~/PAZ/ava-data'),
                        help='Root directory of AVA data (with videos/ and annotations/)')
    parser.add_argument('--target_fps', type=int, default=-1,
                        help='Target FPS for GT alignment/output. Set -1 to auto-match video FPS')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to a specific AVA video file to process')
    parser.add_argument('--architecture', type=str, default='CNN2Plus1D_Light',
                        choices=['VVAD-LRS3-LSTM', 'CNN2Plus1D', 'CNN2Plus1D_Filters',
                                 'CNN2Plus1D_Layers', 'CNN2Plus1D_Light'],
                        help='VVAD model architecture to use')
    parser.add_argument('--stride', type=int, default=10,
                        help='Frames between predictions (higher = fewer predictions)')
    parser.add_argument('--avg', type=int, default=6,
                        help='Averaging window size for predictions (1 disables averaging)')
    parser.add_argument('--average_type', type=str, default='weighted',
                        choices=['mean', 'weighted'],
                        help='Averaging type for predictions')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (default: <video_basename>_vvad.mp4)')
    return parser.parse_args()


def pick_first_annotated_video(dataset: AvaDataset):
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
    # AVA numeric labels from AvaDataset._map_label: 0=NOT_SPEAKING, 1=SPEAKING_BUT_NOT_AUDIBLE, 2=SPEAKING_AND_AUDIBLE
    return 'not-speaking' if ava_numeric_label == 0 else 'speaking'


def run_video_with_gt(video_path: str, output_path: str, dataset: AvaDataset,
                      architecture: str, stride: int,
                      averaging_window_size: int, average_type: str):
    pipeline = dt.DetectVVAD()

    # Prepare frames sampled at dataset.target_fps and ground truth labels
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    annots = dataset._load_annotation_csv(video_name)
    frames_bgr = dataset._extract_frames(video_path)

    if annots == []:
        print('Warning: No annotations found for this video. Running without correctness calculation.')
        return run_video_without_gt(video_path, output_path, architecture, stride, averaging_window_size, average_type)

    # Align labels to frames like AvaDataset._generator does
    ava_numeric_labels = [dataset._map_label(a['label']) for a in annots[:len(frames_bgr)]]
    gt_vvad_labels = [_map_ava_to_vvad_text(v) for v in ava_numeric_labels]

    # Writer uses sampled resolution and dataset.target_fps
    height, width = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, dataset.target_fps, (width, height))

    num_frames = len(frames_bgr)
    num_predictions = 0
    speaking_count = 0
    not_speaking_count = 0
    correct_count = 0

    for idx, frame_bgr in enumerate(frames_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = pipeline(frame_rgb)

        out_rgb = result['image'] if isinstance(result, dict) and 'image' in result else frame_rgb
        pred_label = None

        if isinstance(result, dict) and 'boxes2D' in result and len(result['boxes2D']) > 0:
            box = result['boxes2D'][0]
            pred_label = getattr(box, 'class_name', None)
            if pred_label is not None:
                num_predictions += 1
                if pred_label == 'speaking':
                    speaking_count += 1
                elif pred_label == 'not-speaking':
                    not_speaking_count += 1

        # Compare only when we have a prediction
        if pred_label is not None and idx < len(gt_vvad_labels):
            is_correct = (pred_label == gt_vvad_labels[idx])
            if is_correct:
                correct_count += 1
            # Overlay correctness indicator
            cv2.putText(out_rgb, f"GT: {gt_vvad_labels[idx]} | Pred: {pred_label} | {'OK' if is_correct else 'WRONG'}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if is_correct else (255, 0, 0), 2, cv2.LINE_AA)
        elif idx < len(gt_vvad_labels):
            # Show GT and that no prediction was made
            cv2.putText(out_rgb, f"GT: {gt_vvad_labels[idx]} | Pred: -",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)

        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        writer.write(out_bgr)

    writer.release()

    accuracy = (correct_count / num_predictions) if num_predictions > 0 else None
    print('Processing complete:')
    print(f'  Input:   {video_path}')
    print(f'  Output:  {output_path}')
    print(f'  Sampled frames (for GT): {num_frames} @ {dataset.target_fps} FPS')
    print(f'  Predicted frames: {num_predictions}')
    print(f"  speaking: {speaking_count}, not-speaking: {not_speaking_count}")
    if accuracy is None:
        print('  Accuracy: N/A (no predictions produced)')
    else:
        print(f'  Accuracy: {accuracy:.4f} ({correct_count}/{num_predictions})')


def run_video_without_gt(video_path: str, output_path: str, architecture: str, stride: int,
                         averaging_window_size: int, average_type: str):
    pipeline = dt.DetectVVAD()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 25.0, (width, height))

    num_frames = 0
    num_predictions = 0
    speaking_count = 0
    not_speaking_count = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        num_frames += 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = pipeline(frame_rgb)

        out_rgb = result['image'] if isinstance(result, dict) and 'image' in result else frame_rgb
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        if isinstance(result, dict) and 'boxes2D' in result and len(result['boxes2D']) > 0:
            box = result['boxes2D'][0]
            label = getattr(box, 'class_name', None)
            if label is not None:
                num_predictions += 1
                if label == 'speaking':
                    speaking_count += 1
                elif label == 'not-speaking':
                    not_speaking_count += 1

        writer.write(out_bgr)

    cap.release()
    writer.release()

    print('Processing complete (no GT):')
    print(f'  Input:   {video_path}')
    print(f'  Output:  {output_path}')
    print(f'  Frames:  {num_frames}')
    print(f'  Predicted frames: {num_predictions}')
    print(f"  speaking: {speaking_count}, not-speaking: {not_speaking_count}")


def main():
    args = parse_args()

    # Create dataset first (we may override target_fps after inspecting video)
    dataset = AvaDataset(root_dir=args.root_dir, target_fps=25 if args.target_fps == -1 else args.target_fps)

    if args.video is None:
        video_path = pick_first_annotated_video(dataset)
        if video_path is None:
            raise RuntimeError('No annotated AVA videos found. Ensure annotations exist in root_dir/annotations')
        print(f'No --video provided. Using first annotated video found: {video_path}')
    else:
        video_path = args.video
        if not os.path.isabs(video_path):
            video_path = os.path.join(dataset.video_dir, video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f'Video not found: {video_path}')

    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(os.path.basename(video_path))
        output_path = os.path.join(os.path.dirname(video_path), f"{base}_vvad.mp4")

    # Auto-match dataset target_fps to the video's FPS if requested
    if args.target_fps == -1:
        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        matched_fps = int(round(native_fps)) if native_fps and native_fps > 0 else 25
        dataset.target_fps = max(1, matched_fps)

    run_video_with_gt(
        video_path=video_path,
        output_path=output_path,
        dataset=dataset,
        architecture=args.architecture,
        stride=args.stride,
        averaging_window_size=args.avg,
        average_type=args.average_type,
    )


if __name__ == '__main__':
    main()


