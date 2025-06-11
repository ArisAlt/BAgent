import cv2
import json
import argparse


def replay(log_file: str, delay: int = 500):
    """Display frames from a recorded session with action labels."""
    with open(log_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            frame = cv2.imread(entry['frame'])
            label = entry.get('action', '')
            if frame is None:
                continue
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)
            cv2.imshow('Replay', frame)
            key = cv2.waitKey(delay)
            if key & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Replay recorded session')
    parser.add_argument('--log', type=str, default='recordings/log.jsonl',
                        help='Path to log file')
    parser.add_argument('--delay', type=int, default=500,
                        help='Delay between frames in ms')
    args = parser.parse_args()
    replay(args.log, args.delay)


if __name__ == '__main__':
    main()
