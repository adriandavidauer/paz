import os
import urllib.request
import cv2
import numpy as np
import tensorflow as tf
import logging

class AvaDataset:
    def __init__(self,
                 root_dir="ava_data",
                 file_list_url="https://s3.amazonaws.com/ava-dataset/annotations/ava_speech_file_names_v1.txt",
                 video_url_template="https://s3.amazonaws.com/ava-dataset/trainval/{}",target_fps=25,resize=None, log_level=logging.INFO):
        self.root_dir = root_dir
        self.video_dir = os.path.join(root_dir, "videos")
        self.csv_dir = os.path.join(root_dir, "annotations")
        self.target_fps = target_fps
        self.resize = resize
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        self.file_list_path = os.path.join(self.csv_dir, "ava_speech_file_names_v1.txt")

        self.file_list_url = file_list_url
        self.video_url_template = video_url_template
        self.logger = logging.getLogger("AvaDataset")
        
        self.file_names = self._load_file_list()
        
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

    def _load_file_list(self):
        if not os.path.exists(self.file_list_path):
            self.logger.info("Downloading AVA file list...")
            urllib.request.urlretrieve(self.file_list_url, self.file_list_path)
        with open(self.file_list_path, "r") as f:
            file_names = [line.strip() for line in f if line.strip()]
        self.logger.info(f"Loaded {len(file_names)} video file names.")
        return file_names

    def _download_video(self, file_name):
        local_path = os.path.join(self.video_dir, file_name)
        if os.path.exists(local_path):
            self.logger.info(f"Video already exists: {file_name}")
            return local_path
        url = self.video_url_template.format(file_name)
        self.logger.info(f"Downloading video: {file_name}")
        urllib.request.urlretrieve(url, local_path)
        return local_path

    def _load_annotation_csv(self, video_name):
        csv_path = os.path.join(self.csv_dir, f"{video_name}-activespeaker.csv")
        if not os.path.exists(csv_path):
            self.logger.warning(f"CSV annotations not found for video {csv_path}")
            return []
        annots = []
        with open(csv_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                annots.append({
                    "timestamp": float(parts[1]),
                    "bbox": tuple(map(float, parts[2:6])),
                    "label": parts[6],
                    "entity_id": parts[7]
                })
        return annots

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f'Could not open video file: {video_path}')
        
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        if fps <= 0:
            fps = self.target_fps
            self.logger.warning(f'Could not read FPS from video, using target_fps: {fps}')
        
        interval = max(1, int(fps / self.target_fps))
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.size == 0:
                self.logger.warning(f'Skipping empty frame at count {count}')
                count += 1
                continue
            if count % interval == 0:
                # Keep original resolution unless resize is explicitly specified
                if self.resize is not None and self.resize != (frame.shape[1], frame.shape[0]):
                    frame_resized = cv2.resize(frame, self.resize)
                    frames.append(frame_resized)
                else:
                    frames.append(frame)
            count += 1
        cap.release()
        
        if len(frames) == 0:
            self.logger.warning(f'No frames extracted from video: {video_path}')
        
        return frames

    def _compute_difficulty(self, frames):
        motion_vals = []
        face_counts = []
        prev_gray = None
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                motion_vals.append(np.mean(cv2.absdiff(prev_gray, gray)))
            prev_gray = gray
            if self.face_cascade:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                face_counts.append(len(faces))
        motion_score = float(np.mean(motion_vals)) if motion_vals else 0.0
        face_score = float(np.mean(face_counts)) if face_counts else 0.0
        difficulty = 0.7 * motion_score + 0.3 * face_score
        return {"motion": motion_score, "faces": face_score, "difficulty": difficulty}

    def _generator(self):
        for idx, file_name in enumerate(self.file_names):
            video_name = os.path.splitext(file_name)[0]
            annots = self._load_annotation_csv(video_name)
            if annots == []:
                continue
            video_path = self._download_video(file_name)
            frames = self._extract_frames(video_path)
            #difficulty = self._compute_difficulty(frames)
            labels = np.array([self._map_label(a["label"]) for a in annots[:len(frames)]], dtype=np.int32)
            frames_np = np.stack(frames).astype(np.uint8)
            yield {
                "frames": frames_np,
                "labels": labels,
                #"difficulty": np.float32(difficulty["difficulty"])
            }

    def _map_label(self, label):
        label_map = {
            "SPEAKING_AND_AUDIBLE": 2,
            "SPEAKING_BUT_NOT_AUDIBLE": 1,
            "NOT_SPEAKING": 0
        }
        return label_map.get(label, 0)

    def as_tf_dataset(self, batch_size=2, shuffle=True, num_parallel_calls=tf.data.AUTOTUNE):
        sample_video = self._generator().__next__()
        num_frames, h, w, _ = sample_video["frames"].shape

        output_signature = {
            "frames": tf.TensorSpec(shape=(num_frames, h, w, 3), dtype=tf.uint8),
            "labels": tf.TensorSpec(shape=(num_frames,), dtype=tf.int32),
            "difficulty": tf.TensorSpec(shape=(), dtype=tf.float32)
        }

        ds = tf.data.Dataset.from_generator(lambda: self._generator(), output_signature=output_signature)

        if shuffle:
            ds = ds.shuffle(buffer_size=4)
        ds = ds.map(lambda x: {"frames": tf.image.convert_image_dtype(x["frames"], tf.float32),
                               "labels": x["labels"],
                               #"difficulty": x["difficulty"]
                              },
                    num_parallel_calls=num_parallel_calls)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def __len__(self):
        return len(self.file_names)
    
    def download_all_videos(self):
        for file_name in self.file_names:
            self._download_video(file_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download AVA Dataset Videos")
    parser.add_argument("--root_dir", type=str, default="ava_data", help="Root directory for AVA dataset")
    args = parser.parse_args()
    dataset = AvaDataset(root_dir=args.root_dir)
    dataset.download_all_videos()
