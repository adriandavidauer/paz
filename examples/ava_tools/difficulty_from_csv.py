import csv
from collections import defaultdict
from statistics import mean

def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    areaA = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    areaB = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = areaA + areaB - inter + 1e-8
    return inter / union

def load_ava_as(csv_path):
    """Load Active Speaker CSV annotations.

    # Arguments
        csv_path: String. Path to a single AVA CSV file.

    # Returns
        Dictionary mapping video_id → list of annotation dictionaries.
        Each annotation contains:
            - ts: Float timestamp
            - box: Tuple (x1, y1, x2, y2)
            - label: String
            - ent: String entity id
    """

    by_vid = defaultdict(list)
    with open(csv_path, newline='') as f:
        r = csv.reader(f)
        for row in r:
            vid, ts = row[0], float(row[1])
            x1, y1, x2, y2 = map(float, row[2:6])
            label = row[6].strip()
            ent = row[7].strip()
            by_vid[vid].append({
                'ts': ts, 'box': (x1, y1, x2, y2),
                'label': label, 'ent': ent
            })
    for vid in by_vid:
        by_vid[vid].sort(key=lambda d: (d['ts'], d['ent']))
    return by_vid

def compute_difficulty(rows):
    """Compute difficulty metrics for one video.

    # Arguments
        rows: List of dictionaries. Each contains timestamp, bounding box,
              label, and entity ID for that video.

    # Returns
        Dictionary with:
            - diversity: Int. Count of unique entity IDs.
            - interactivity_avg: Float. Avg faces per frame.
            - interactivity_max: Int. Max faces per frame.
            - dynamics_mean: Float. Mean motion (1 - IoU).
            - dynamics_var: Float. Variance of motion.
            - occlusion: Float. Avg pairwise IoU per frame.
            - audibility_sna_share: Float. Ratio of NOT_AUDIBLE speech.
    """

    # group by timestamp
    by_ts = defaultdict(list)
    entities = set()
    for r in rows:
        by_ts[r['ts']].append(r)
        entities.add(r['ent'])

    # interactivity
    faces_per_frame = [len(v) for v in by_ts.values()]
    interactivity_avg = mean(faces_per_frame) if faces_per_frame else 0.0
    interactivity_max = max(faces_per_frame) if faces_per_frame else 0

    # diversity
    diversity = len(entities)

    # audibility: share of SPEAKING_NOT_AUDIBLE among speaking frames
    speak = [r for r in rows if r['label'].startswith('SPEAKING')]
    sna = [r for r in speak if 'NOT_AUDIBLE' in r['label']]
    audibility_sna_share = (len(sna) / len(speak)) if speak else 0.0

    # dynamics: mean (1 - IoU) per entity over successive frames
    by_ent = defaultdict(list)
    for r in rows:
        by_ent[r['ent']].append(r)
    dynamics_terms = []
    for ent, seq in by_ent.items():
        seq.sort(key=lambda d: d['ts'])
        for i in range(1, len(seq)):
            iou_val = iou(seq[i-1]['box'], seq[i]['box'])
            dynamics_terms.append(1.0 - iou_val)
    dynamics = mean(dynamics_terms) if dynamics_terms else 0.0

    # occlusion: average pairwise IoU per frame when >=2 faces
    occlusion_terms = []
    for ts, dets in by_ts.items():
        n = len(dets)
        if n < 2:
            continue
        boxes = [d['box'] for d in dets]
        # average over all pairs
        s, c = 0.0, 0
        for i in range(n):
            for j in range(i+1, n):
                s += iou(boxes[i], boxes[j])
                c += 1
        if c:
            occlusion_terms.append(s / c)
    occlusion = mean(occlusion_terms) if occlusion_terms else 0.0

    return dict(
        diversity=diversity,
        interactivity_avg=interactivity_avg,
        interactivity_max=interactivity_max,
        audibility_sna_share=audibility_sna_share,
        occlusion=occlusion,
        dynamics=dynamics
    )

if __name__ == "__main__":
    import sys, os, json, glob
    path = sys.argv[1]  

    out = {}

    def process_csv(csv_path):
        by_vid = load_ava_as(csv_path)
        for vid, rows in by_vid.items():
            out[vid] = compute_difficulty(rows)

    if os.path.isdir(path):
        # Process every *.csv in the folder
        for csv_path in sorted(glob.glob(os.path.join(path, "*.csv"))):
            process_csv(csv_path)

    # Save JSON next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(script_dir, "difficulty_ava_test.json")

    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved difficulty metrics to: {out_file}")



