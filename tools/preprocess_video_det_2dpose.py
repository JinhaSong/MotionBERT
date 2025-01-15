import os
import sys
import cv2
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.cfg import load_cfg
from lib.yolo.yolov7.det import YOLOv7
from lib.pose.alphapose.pose import AlphaPose

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--device", type=str, default="0", help="GPU device number")
    parser.add_argument("--det-cfg", type=str, default="configs/det/yolov7x.yml", help="Path of object detection parameter")
    parser.add_argument("--pose-cfg", type=str, default="configs/pose/alphapose.yml", help="Path of pose estimation parameter")
    parser.add_argument("--video-path", required=True, help="Path to the dataset directory")
    opt = parser.parse_known_args()[0]

    device = opt.device
    det_cfg = load_cfg(opt.det_cfg)["infer"]
    pose_cfg = load_cfg(opt.pose_cfg)["infer"]
    video_path = opt.video_path

    pose_model = AlphaPose(pose_cfg, device)
    print(f"Pose estimation model({pose_cfg['model_name']}) is loaded.")

    det_model = YOLOv7(det_cfg, device)
    print(f"Object detection model({det_cfg['model_name']}) is loaded.")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print("Error: Video contains no frames.")
        cap.release()

    middle_frame_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)

    ret, frame = cap.read()

    if not ret:
        print(f"Error: Unable to read frame at index {middle_frame_idx}.")
        cap.release()

    det = det_model.inference(frame)
    pose = pose_model.inference_image(frame, det, "0.jpg")
    print(pose, len(pose[0]['keypoints']))