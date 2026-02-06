#!/usr/bin/env python3
"""Enroll a user via the web interface.

This script helps enroll a user's face for recognition.
Run on the RunPod server.

Usage:
    python scripts/enroll_user.py --name "Your Name" --id "your_id"
"""

import argparse
import sys
import time

from covision import FaceRecognizer, Camera


def main():
    parser = argparse.ArgumentParser(description="Enroll a user for face recognition")
    parser.add_argument("--name", required=True, help="Display name")
    parser.add_argument("--id", help="User ID (defaults to name)")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames")
    parser.add_argument("--camera", type=int, default=0, help="Camera device")

    args = parser.parse_args()

    user_id = args.id or args.name.lower().replace(" ", "_")

    print(f"Enrolling: {args.name} (ID: {user_id})")
    print(f"Capturing {args.frames} frames...")
    print()

    # Initialize recognizer
    recognizer = FaceRecognizer(
        embeddings_path="models/embeddings",
    )
    recognizer.load()

    # Initialize camera
    camera = Camera(device=args.camera)
    camera.start()

    print("Look at the camera and move your head slightly...")
    time.sleep(2)

    # Capture frames
    frames = []
    for i in range(args.frames * 3):
        frame_data = camera.read()
        if frame_data:
            frames.append(frame_data.frame)
            print(f"  Captured frame {len(frames)}/{args.frames}")

        if len(frames) >= args.frames:
            break

        time.sleep(0.2)

    camera.stop()

    if len(frames) < 3:
        print("ERROR: Not enough frames captured")
        sys.exit(1)

    # Enroll
    success = recognizer.enroll(user_id, args.name, frames)

    if success:
        print()
        print(f"Successfully enrolled {args.name}!")
        print(f"Embeddings saved to: models/embeddings/{user_id}.npz")
    else:
        print()
        print("Enrollment failed. Make sure your face is visible.")
        sys.exit(1)


if __name__ == "__main__":
    main()
