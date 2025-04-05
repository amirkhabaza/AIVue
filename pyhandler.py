import sys
import os
import subprocess
import psutil
import time
import cv2

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QMessageBox,
    QSizePolicy
)
from PyQt6.QtCore import (QProcess, Qt)

from pipefacemac import HeadGazeTracker

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPEFACE_SCRIPT = os.path.join(SCRIPT_DIR, "pipefacemac.py")    #CHANGE PATH
PYGUI_SCRIPT = os.path.join(SCRIPT_DIR, "pygui.py")             #CHANGE PATH
PYTHON_EXECUTABLE = sys.executable
# --------------------


class ProcessControlApp(QWidget):
    """Main application window for controlling external Python scripts."""
    def __init__(self):
        self.is_recording = False  # Add state flag
        super().__init__()
        self.pipeface_process: psutil.Process | None = None # psutil handle for pipefacemac.py
        self.is_paused = False
        self.init_ui()
        self._start_pipeface()

    def _handle_talk(self):  # Renamed from _handle_placeholder
        self.is_recording = not self.is_recording
        print(f"Recording {'STARTED' if self.is_recording else 'STOPPED'}")

    def init_ui(self):
    # Remove window frame and keep always on top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )

    # Optional: make background translucent if needed
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.btn_talk = QPushButton("Talk")
        self.btn_talk.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
       
        self.btn_assist = QPushButton("Assist")
        self.btn_assist.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.btn_recall = QPushButton("Recall")
        self.btn_recall.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.btn_quit = QPushButton("Quit")
        self.btn_quit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
       

        # Connect signals - Talk/Assist have placeholders
        self.btn_talk.clicked.connect(self._handle_placeholder)
        self.btn_assist.clicked.connect(self._handle_placeholder)
        self.btn_pause.clicked.connect(self._toggle_pause_resume)
        self.btn_recall.clicked.connect(self._recall_pipeface)
        self.btn_quit.clicked.connect(self._quit_all)

        layout.addWidget(self.btn_talk)
        layout.addWidget(self.btn_assist)
        layout.addWidget(self.btn_pause)
        layout.addWidget(self.btn_recall)
        layout.addWidget(self.btn_quit)

        self.setLayout(layout)
        self._position_on_right()
        self.show()
    
    def _position_on_right(self):
        screen = QApplication.primaryScreen().availableGeometry()
        window_width = 100  # You can change this value
        window_height = screen.height()

        self.setGeometry(
            screen.width() - window_width,  # Right side
            0,                              # Top
            window_width,
            window_height
        )

    def _handle_placeholder(self):
        sender = self.sender()
        print(f"Button '{sender.text()}' clicked (no action defined).")

    def _is_pipeface_running(self) -> bool:
        """Checks if the managed pipefacemac.py process is running."""
        if self.pipeface_process:
            try:
                return self.pipeface_process.is_running()
            except psutil.NoSuchProcess:
                self.pipeface_process = None
                return False
            except Exception as e:
                print(f"Error checking process status: {e}")
                return False
        return False

    def _start_pipeface(self) -> bool:
        """Starts or restarts the pipefacemac.py script using psutil."""
        if not os.path.exists(PIPEFACE_SCRIPT):
            print(f"Error: Script not found at {PIPEFACE_SCRIPT}")
            QMessageBox.warning(self, "Error", f"Script not found:\n{PIPEFACE_SCRIPT}")
            return False

        if self._is_pipeface_running():
            print("Terminating existing pipefacemac.py before restart.")
            self._terminate_process(self.pipeface_process)

        try:
            print(f"Starting script: {PIPEFACE_SCRIPT}")
            # Use creationflags on Windows if console window needs hiding (adjust if needed)
            kwargs = {}
            # if sys.platform == "win32": kwargs['creationflags'] = 0x08000000 # CREATE_NO_WINDOW

            process_popen = subprocess.Popen([PYTHON_EXECUTABLE, PIPEFACE_SCRIPT], **kwargs)
            self.pipeface_process = psutil.Process(process_popen.pid)
            self.is_paused = False
            self.btn_pause.setText("Pause")
            print(f"Started pipefacemac.py with PID: {self.pipeface_process.pid}")
            return True
        except Exception as e:
            print(f"Error starting {PIPEFACE_SCRIPT}: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start script:\n{PIPEFACE_SCRIPT}\n\n{e}")
            self.pipeface_process = None
            return False

    def _toggle_pause_resume(self):
        """Pauses or resumes the pipefacemac.py script via psutil."""
        if not self._is_pipeface_running():
            print("Cannot pause/resume: pipefacemac.py is not running.")
            return

        try:
            if self.is_paused:
                print(f"Resuming PID {self.pipeface_process.pid}...")
                self.pipeface_process.resume()
                self.btn_pause.setText("Pause")
                self.is_paused = False
            else:
                print(f"Pausing PID {self.pipeface_process.pid}...")
                self.pipeface_process.suspend()
                self.btn_pause.setText("Resume")
                self.is_paused = True
        except psutil.Error as e:
            print(f"Error pausing/resuming process {self.pipeface_process.pid}: {e}")
            QMessageBox.warning(self, "Error", f"Failed to pause/resume process:\n{e}")
            # Attempt to revert state on error
            self.is_paused = not self.is_paused
            self.btn_pause.setText("Pause" if not self.is_paused else "Resume")

    def _recall_pipeface(self):
        """Terminates and restarts the pipefacemac.py script."""
        print("Recall requested...")
        if self._is_pipeface_running():
            print("Terminating existing pipefacemac.py process...")
            if self._terminate_process(self.pipeface_process):
                 time.sleep(0.5) # Brief pause before restart
            else:
                 print("Failed to terminate existing process cleanly, attempting restart anyway.")
        else:
            print("pipefacemac.py was not running. Starting it now.")

        self._start_pipeface()

    def _terminate_process(self, process: psutil.Process | None) -> bool:
        """Attempts to terminate a process gracefully (terminate -> wait -> kill)."""
        if not process: return False
        try:
            if process.is_running():
                print(f"Terminating process PID {process.pid}...")
                process.terminate()
                try:
                    process.wait(timeout=2)
                    print(f"Process {process.pid} terminated.")
                    return True
                except psutil.TimeoutExpired:
                    print(f"Process {process.pid} kill required...")
                    process.kill()
                    process.wait(timeout=1)
                    print(f"Process {process.pid} killed.")
                    return True
        except psutil.NoSuchProcess:
            print(f"Process {process.pid} already terminated.")
            return True
        except psutil.Error as e:
            print(f"Error terminating process {process.pid}: {e}")
            return False
        finally:
            if process == self.pipeface_process:
                 self.pipeface_process = None # Clear handle if it was the managed one

    def _find_and_kill_processes(self, script_full_paths: list[str]):
        """Finds running python processes matching script paths and kills them."""
        killed_count = 0
        target_scripts = [os.path.normpath(p) for p in script_full_paths]
        print(f"Attempting to find and kill processes for: {target_scripts}")

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline')
                # Check if it's a python process running one of the target scripts
                if cmdline and len(cmdline) > 1 and sys.executable in cmdline[0]:
                    script_path = os.path.normpath(cmdline[1])
                    if script_path in target_scripts:
                         print(f"Found target process: PID={proc.pid}, Script={script_path}")
                         if self._terminate_process(psutil.Process(proc.pid)):
                             killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue # Ignore processes that vanished or are inaccessible
            except Exception as e:
                print(f"Error iterating processes: {e}") # Log other unexpected errors
        print(f"Killed {killed_count} target process(es).")

    def _quit_all(self):
        """Terminates target scripts and closes the application."""
        print("Quit requested. Terminating target processes...")

        # Terminate the specific instance started by this app
        self._terminate_process(self.pipeface_process)

        # Find and kill any other instances of target scripts
        tracker.stop_tracking()  # Signal the tracker to stop
        # tracker_thread.join()    # Wait for the tracker to finish cleanup
        scripts_to_kill = []
        if os.path.exists(PIPEFACE_SCRIPT): scripts_to_kill.append(PIPEFACE_SCRIPT)
        if os.path.exists(PYGUI_SCRIPT): scripts_to_kill.append(PYGUI_SCRIPT)
        cv2.destroyAllWindows()
        

        if scripts_to_kill:
             self._find_and_kill_processes(scripts_to_kill)
        else:
             print("No target script paths found to search for.")

        print("Closing controller application.")
        self.close() # Close this PyQt application

    def closeEvent(self, event):
        """Ensures cleanup when the window is closed via 'X'."""
        print("Close event triggered. Cleaning up...")
        # Run termination logic on window close as well
        self._terminate_process(self.pipeface_process)
        scripts_to_kill = []
        if os.path.exists(PIPEFACE_SCRIPT): scripts_to_kill.append(PIPEFACE_SCRIPT)
        if os.path.exists(PYGUI_SCRIPT): scripts_to_kill.append(PYGUI_SCRIPT)
        if scripts_to_kill: self._find_and_kill_processes(scripts_to_kill)
        # cv2.destroyAllWindows()

        event.accept()


if __name__ == '__main__':
    # Optional: Basic check if the primary controlled script exists
    if not os.path.exists(PIPEFACE_SCRIPT):
         print(f"WARNING: Script 'pipefacemac.py' not found at {PIPEFACE_SCRIPT}. Recall/Pause will fail.")
         # Consider exiting if critical:
         # QMessageBox.critical(None, "Startup Error", f"Required script 'pipefacemac.py' not found.\n{PIPEFACE_SCRIPT}")
         # sys.exit(1)

    app = QApplication(sys.argv)
    controller = ProcessControlApp()
    sys.exit(app.exec())
