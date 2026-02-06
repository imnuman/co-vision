"""Demo script for CoVision.

Run with: python -m covision.demo
"""

import logging
import sys

from covision import VisionSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run the CoVision demo."""
    print("=" * 60)
    print("CoVision Demo")
    print("=" * 60)
    print()
    print("This demo will:")
    print("  1. Start the webcam")
    print("  2. Detect persons in frame")
    print("  3. Recognize enrolled faces")
    print("  4. Track gaze/attention")
    print("  5. Emit events when you arrive/leave/look at camera")
    print()
    print("Press Ctrl+C to stop")
    print()

    # Create vision system
    vision = VisionSystem()

    # Register event handlers
    @vision.on("user_arrived")
    def on_arrival(event):
        print(f"\n>>> USER ARRIVED (confidence={event.confidence:.2f})")

    @vision.on("user_left")
    def on_departure(event):
        print(f"\n>>> USER LEFT (was here for {event.duration_seconds:.1f}s)")

    @vision.on("user_looking")
    def on_looking(event):
        print("\n>>> USER IS LOOKING AT CAMERA")

    @vision.on("scene_update")
    def on_scene(event):
        print(f"\n>>> SCENE: {event.description}")

    # Start the vision system
    try:
        vision.start()
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

    print("\nDemo finished.")


if __name__ == "__main__":
    main()
