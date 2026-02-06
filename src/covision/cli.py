"""Command-line interface for CoVision."""

import argparse
import logging
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="covision",
        description="CoVision - Computer Vision for AI Companions",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run the demo")

    # Enroll command
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a new user")
    enroll_parser.add_argument("--name", required=True, help="User display name")
    enroll_parser.add_argument("--id", help="User ID (defaults to name)")
    enroll_parser.add_argument(
        "--frames", type=int, default=10,
        help="Number of frames to capture",
    )

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify enrollment")
    verify_parser.add_argument("--name", required=True, help="User name to verify")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmark")
    benchmark_parser.add_argument(
        "--iterations", type=int, default=100,
        help="Number of iterations",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "demo":
        from covision.demo import main as demo_main
        demo_main()

    elif args.command == "enroll":
        from covision import VisionSystem
        vision = VisionSystem()
        vision._init_components()
        vision.camera.start()

        user_id = args.id or args.name.lower().replace(" ", "_")
        print(f"Enrolling user: {args.name} (ID: {user_id})")
        print("Look at the camera...")

        import time
        time.sleep(2)  # Give user time to position

        success = vision.enroll_user(user_id, args.name, args.frames)

        vision.camera.stop()

        if success:
            print(f"Successfully enrolled {args.name}!")
        else:
            print("Enrollment failed. Make sure your face is visible.")
            sys.exit(1)

    elif args.command == "verify":
        from covision import VisionSystem, FaceRecognizer
        vision = VisionSystem()
        vision._init_components()

        user_id = args.name.lower().replace(" ", "_")
        user = vision.recognizer.get_user(user_id)

        if user:
            print(f"User found: {user.name}")
            print(f"  Embeddings: {len(user.embeddings)}")
        else:
            print(f"User not found: {args.name}")
            sys.exit(1)

    elif args.command == "benchmark":
        print("Benchmark not yet implemented")
        # TODO: Implement benchmark

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
