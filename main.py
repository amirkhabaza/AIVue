import platform

if platform.system() == "Darwin":
    from pipefacemac import HeadGazeTracker
elif platform.system() == "Windows":
    from action_executor import HeadGazeTracker  # Adjust if needed
else:
    from element_detector_demo import HeadGazeTracker  # Fallback for other OSes

def main():
    tracker = HeadGazeTracker()
    tracker.start_tracking()

if __name__ == "__main__":
    main()