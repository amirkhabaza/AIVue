import tkinter as tk
import vosk
import sounddevice as sd
import queue
import threading
import json
import pyautogui
import sys
import os
import time # Added for potential small delays if needed

# --- Platform Specific Imports ---
IS_WINDOWS = False # Default to False
try:
    # Needed for WS_EX_NOACTIVATE on Windows
    import win32gui
    import win32con
    import ctypes # Needed for GetWindowLongPtr / SetWindowLongPtr on 64-bit Python
    IS_WINDOWS = True
except ImportError:
    # Keep IS_WINDOWS as False
    print("-----------------------------------------------------------")
    print("WARNING: 'pywin32' not found or not running on Windows.")
    print("The overlay window *will* steal focus when buttons are clicked.")
    print("Install 'pywin32' for focus prevention: pip install pywin32")
    print("-----------------------------------------------------------")


# --- Configuration ---
MODEL_PATH = "E:/Code/vosk/vosk-model-en-us-0.42-gigaspeech"
SAMPLE_RATE = 16000
DEVICE_ID = None
BLOCK_SIZE = 8000
BUTTON_WIDTH = 80
BUTTON_HEIGHT = 40
TOTAL_WINDOW_HEIGHT = BUTTON_HEIGHT * 2
BUTTON_PADDING = 10

# --- Global Variables ---
q = queue.Queue()
result_queue = queue.Queue()
is_recording = False
stream = None
recognizer = None
model = None
processing_thread = None
typing_thread = None

# --- Vosk Model Loading ---
def load_vosk_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Vosk model folder not found at '{MODEL_PATH}'")
        print("Please download a model from https://alphacephei.com/vosk/models")
        print("and update the MODEL_PATH variable in the script.")
        return False
    elif MODEL_PATH == "path/to/your/vosk-model-folder":
         print("\n\n*** WARNING: Using default MODEL_PATH placeholder. ***")
         print("*** Please edit the script and set the 'MODEL_PATH' variable ***\n\n")
         # Optionally return False if path is not set correctly
         # return False

    try:
        print(f"Loading Vosk model from: {MODEL_PATH}")
        model = vosk.Model(MODEL_PATH)
        print("Vosk model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading Vosk model: {e}")
        print("Make sure the path is correct and the model files are intact.")
        return False

# --- Audio Callback ---
def audio_callback(indata, frames, time, status):
    if status:
        # Suppress potential harmless overflow warnings if needed
        # if "input overflow" not in str(status).lower():
        print(status, file=sys.stderr)
    if is_recording: # Only queue data if we are actively recording
        try:
            q.put(bytes(indata))
        except Exception as e:
            print(f"Error putting data into queue: {e}") # Should not happen often

# --- Audio Processing Thread ---
def process_audio():
    global recognizer, is_recording # Use the global recognizer

    if recognizer is None:
         print("ERROR: Recognizer not initialized before processing thread start.")
         result_queue.put(None) # Signal typing thread to stop
         return

    print("Audio processing thread started (using existing recognizer).")
    while is_recording:
        try:
            data = q.get(timeout=0.1) # Wait max 0.1s for data

            # Check again if recognizer became None (e.g., during stop)
            if not is_recording or recognizer is None:
                 break

            if recognizer.AcceptWaveform(data):
                result_json = recognizer.Result()
                result_dict = json.loads(result_json)
                # Get text and strip leading/trailing whitespace
                final_text = result_dict.get('text', '').strip()

                # <<< --- FILTERING LOGIC for "the" --- >>>
                is_only_the = final_text.lower() == "the"

                if final_text and not is_only_the: # Only queue if not empty AND not just "the"
                    result_queue.put(("final", final_text))
                    print(f"Final: {final_text}")
                elif is_only_the:
                    print(f"Filtered out spurious 'the'. Original JSON: {result_json}") # Debugging info
                # <<< --- END FILTERING LOGIC --- >>>

            else:
                partial_json = recognizer.PartialResult()
                partial_dict = json.loads(partial_json)
                partial_text = partial_dict.get('partial', '')
                if partial_text:
                    # Avoid flooding with very rapid partials if needed
                    # Consider adding a small delay or threshold here
                    result_queue.put(("partial", partial_text))
                    # Reduce frequency of partial prints if too noisy
                    # print(f"Partial: {partial_text}") # Debug (can be verbose)

        except queue.Empty:
            continue # No data, just loop again if still recording
        except json.JSONDecodeError as e:
             print(f"Error decoding JSON from Vosk: {e} - JSON was: {result_json if 'result_json' in locals() else partial_json if 'partial_json' in locals() else 'N/A'}")
             continue # Skip this result
        except Exception as e:
            print(f"Error in audio processing loop: {e}")
            # Consider more specific error handling if needed
            break # Exit thread on significant error

    # Process any remaining audio after stop signal (is_recording is False)
    print("Processing final audio chunk...")
    try:
        # Ensure recognizer still exists before calling FinalResult
        if recognizer:
            final_result_json = recognizer.FinalResult()
            final_result_dict = json.loads(final_result_json)
            final_text = final_result_dict.get('text', '').strip()

            # <<< --- FILTERING LOGIC (Apply again for FinalResult) --- >>>
            is_only_the = final_text.lower() == "the"

            if final_text and not is_only_the:
                result_queue.put(("final", final_text))
                print(f"Final (at end): {final_text}")
            elif is_only_the:
                print(f"Filtered out final 'the' (at end). Original JSON: {final_result_json}") # Debugging info
            # <<< --- END FILTERING LOGIC --- >>>
        else:
            print("Recognizer was None during final processing.")

    except json.JSONDecodeError as e:
        print(f"Error decoding final JSON from Vosk: {e} - JSON was: {final_result_json if 'final_result_json' in locals() else 'N/A'}")
    except Exception as e:
        print(f"Error getting final result: {e}")

    print("Audio processing thread finished.")
    result_queue.put(None) # Signal typing thread to stop


# --- Typing Thread ---
def type_results():
    global is_recording
    print("Typing thread started.")
    current_segment_text = ""

    while True:
        try:
            item = result_queue.get() # Blocking get
            if item is None:
                # Ensure any remaining partial text is finalized if needed
                # (Although Vosk's final result should handle this better)
                if current_segment_text:
                     pyautogui.write(' ') # Add space after last partial segment
                     print("Typing thread adding final space.")
                     current_segment_text = "" # Reset
                break # Exit signal received

            type_flag, text = item

            if not text and type_flag != "final":
                 continue # Skip empty partials

            # Ensure we don't type if recording has just been stopped
            # (Helps prevent race conditions where a late result gets typed)
            # if not is_recording and type_flag != "final": # Allow final result even if stopped
            #      print("Skipping type, recording stopped.")
            #      continue

            if type_flag == "partial":
                # Logic to handle backspacing and updating partial results
                # This basic version just appends; more complex logic could replace
                if text.startswith(current_segment_text):
                    new_chars = text[len(current_segment_text):]
                    if new_chars:
                        pyautogui.write(new_chars, interval=0.01) # Small interval might help reliability
                        current_segment_text = text
                else:
                    # Handle cases where partial result diverges significantly
                    # Option 1: Backspace and rewrite (can be visually disruptive)
                    # backspaces = len(current_segment_text)
                    # if backspaces > 0:
                    #     pyautogui.press('backspace', presses=backspaces, interval=0.01)
                    # pyautogui.write(text, interval=0.01)
                    # current_segment_text = text

                    # Option 2: Just append (simpler, might lead to odd output if vosk jumps)
                     if len(text) > len(current_segment_text): # basic append if longer
                         new_chars = text[len(current_segment_text):]
                         pyautogui.write(new_chars, interval=0.01)
                         current_segment_text = text
                     # else: Ignore shorter/different partials for now


            elif type_flag == "final":
                 # Type any remaining part of the final text not covered by partials
                 if text.startswith(current_segment_text):
                     new_chars = text[len(current_segment_text):]
                     if new_chars:
                         pyautogui.write(new_chars, interval=0.01)
                 elif current_segment_text: # If partial was different, maybe backspace?
                      # Or just type the final result cleanly (might be better)
                      # backspaces = len(current_segment_text)
                      # pyautogui.press('backspace', presses=backspaces, interval=0.01)
                      # pyautogui.write(text, interval=0.01)
                      # For simplicity, let's just append the space or full text if no partial
                      pass # Handled below
                 elif text: # No partials received for this segment
                    pyautogui.write(text, interval=0.01)


                 # Add a space after every final recognized segment
                 pyautogui.write(' ', interval=0.01)
                 current_segment_text = "" # Reset for the next segment

        except queue.Empty:
            # This shouldn't happen with a blocking get unless None is received
            print("Typing thread queue unexpectedly empty.")
            time.sleep(0.05) # Small sleep before retry
            continue
        except Exception as e:
            print(f"Error in typing thread: {e}")
            # Consider logging the error more formally
            break # Exit thread on error

    print("Typing thread finished.")


# --- Button Actions ---
def toggle_recording():
    global is_recording, stream, processing_thread, typing_thread, recognizer, model

    if not is_recording:
        # --- START RECORDING ---
        if not model:
             print("ERROR: Vosk model not loaded. Cannot start recording.")
             # Optionally show a message box
             # tk.messagebox.showerror("Error", "Vosk model not loaded.")
             return

        try:
            # Clear queues before starting
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break
            while not result_queue.empty():
                try: result_queue.get_nowait()
                except queue.Empty: break

            print("Creating new KaldiRecognizer instance.")
            # Recreate recognizer for a clean state
            recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
            recognizer.SetWords(False) # Optional: Disable word timestamps if not needed

            is_recording = True # Set flag *before* starting threads/stream

            try:
                 sd.check_input_settings(device=DEVICE_ID, samplerate=SAMPLE_RATE, channels=1)
            except Exception as e:
                 print(f"Error checking input device settings: {e}")
                 print("Please check if microphone is connected and permissions are granted.")
                 # tk.messagebox.showerror("Audio Error", f"Cannot access microphone: {e}")
                 is_recording = False # Reset flag
                 recognizer = None # Clean up recognizer
                 update_button_state() # Update UI
                 return

            # Start the audio stream
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                device=DEVICE_ID,
                channels=1,
                dtype='int16',
                callback=audio_callback,
                # latency='low' # Can try 'low' latency if needed, might increase CPU
            )
            stream.start()
            print("Recording started...")

            # Start processing and typing threads
            # Make sure they use the *new* recognizer instance
            processing_thread = threading.Thread(target=process_audio, daemon=True)
            typing_thread = threading.Thread(target=type_results, daemon=True)
            processing_thread.start()
            typing_thread.start()

            update_button_state() # Update Record/Stop button state AFTER starting

        except Exception as e:
            print(f"Error starting recording: {e}")
            # tk.messagebox.showerror("Start Error", f"Could not start recording: {e}")
            is_recording = False # Ensure flag is False on error
            if stream and stream.active:
                try: stream.stop(); stream.close()
                except Exception as e_close: print(f"Error closing stream during start error: {e_close}")
            stream = None
            recognizer = None
            # Ensure threads that might have started are handled (though likely didn't get far)
            # (Joining might be complex here, focus on cleanup)
            update_button_state() # Update UI

    else:
        # --- STOP RECORDING ---
        print("Stopping recording...")
        is_recording = False # Signal threads to stop *first*

        # Stop and close the audio stream quickly
        if stream:
            try:
                if stream.active:
                    stream.stop()
                    print("Audio stream stopped.")
                stream.close()
                print("Audio stream closed.")
            except Exception as e_close:
                print(f"Error stopping/closing stream: {e_close}")
            finally:
                stream = None # Ensure stream is marked as closed

        # Wait for threads to finish processing remaining data
        # The process_audio thread will put None in result_queue when done
        # The type_results thread will exit when it gets None
        if processing_thread and processing_thread.is_alive():
             print("Waiting for processing thread to finish...")
             processing_thread.join(timeout=2.0) # Increased timeout slightly
             if processing_thread.is_alive():
                  print("Warning: Processing thread did not finish cleanly.")
        if typing_thread and typing_thread.is_alive():
              print("Waiting for typing thread to finish...")
              typing_thread.join(timeout=1.5) # Increased timeout slightly
              if typing_thread.is_alive():
                   print("Warning: Typing thread did not finish cleanly.")

        # Clean up recognizer *after* threads are done
        # (process_audio needs it until the very end)
        print("Cleaning up recognizer instance.")
        recognizer = None

        update_button_state() # Update Record/Stop button state
        print("Recording stopped.")


def press_enter_key():
    """Simulates pressing the Enter key. Focus prevention is handled by the window style."""
    print("Simulating Enter key press.")
    try:
        # Small delay *might* sometimes help ensure the OS has processed
        # the button click fully before sending the key event, but often not needed.
        # time.sleep(0.05)
        pyautogui.press('enter')
    except Exception as e:
        print(f"Error pressing Enter key: {e}")


# --- UI Setup ---
root = tk.Tk()
root.title("STT Ctrl") # Shorter title
root.geometry(f"{BUTTON_WIDTH}x{TOTAL_WINDOW_HEIGHT}")
root.resizable(False, False)
root.overrideredirect(True) # Makes it borderless
root.attributes('-topmost', True) # Keep it on top

# Position window at top right
screen_width = root.winfo_screenwidth()
x_pos = screen_width - BUTTON_WIDTH - BUTTON_PADDING
y_pos = BUTTON_PADDING
root.geometry(f'+{x_pos}+{y_pos}')

# --- Create Buttons ---
# Record/Stop Button (Top)
# Set takefocus=0 although WS_EX_NOACTIVATE is the main mechanism on Windows
record_button = tk.Button(root, text="Record", command=toggle_recording,
                          bg="green", fg="white",
                          font=("Arial", 10, "bold"),
                          relief=tk.RAISED, state=tk.DISABLED, takefocus=0)
record_button.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

# Enter Button (Bottom)
# Set takefocus=0
enter_button = tk.Button(root, text="Enter", command=press_enter_key,
                         bg="blue", fg="white",
                         font=("Arial", 10, "bold"),
                         relief=tk.RAISED, takefocus=0)
enter_button.pack(side=tk.BOTTOM, expand=True, fill=tk.BOTH)

def update_button_state():
    """Changes Record/Stop button appearance based on recording state."""
    # Check if root window exists before configuring (prevents error on exit)
    if not root.winfo_exists():
        return
    global record_button
    if is_recording:
        record_button.config(text="Stop", bg="red", state=tk.NORMAL)
    else:
        # Only enable record button if the model is loaded
        state = tk.NORMAL if model else tk.DISABLED
        record_button.config(text="Record", bg="green", state=state)
    # Ensure UI updates immediately
    root.update_idletasks()


# --- Apply Window Styles (Windows Specific) ---
def prevent_focus_steal(tk_window):
    """Applies WS_EX_NOACTIVATE style to prevent focus stealing on Windows."""
    if not IS_WINDOWS:
        print("Focus prevention skipped (not on Windows or pywin32 missing).")
        return

    try:
        hwnd = tk_window.winfo_id()
        # Get current extended window style
        # Use GetWindowLongPtr for 64-bit Python compatibility
        if ctypes.sizeof(ctypes.c_void_p) == 8: # 64-bit Python
            style = win32gui.GetWindowLongPtr(hwnd, win32con.GWL_EXSTYLE)
        else: # 32-bit Python
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)

        # Add the WS_EX_NOACTIVATE style using bitwise OR
        style |= win32con.WS_EX_NOACTIVATE

        # Set the new extended window style
        # Use SetWindowLongPtr for 64-bit Python compatibility
        if ctypes.sizeof(ctypes.c_void_p) == 8: # 64-bit Python
            win32gui.SetWindowLongPtr(hwnd, win32con.GWL_EXSTYLE, style)
        else: # 32-bit Python
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)

        print("Applied WS_EX_NOACTIVATE style to prevent focus stealing.")
        # Optional: Force redraw/update if needed, but usually not necessary
        # win32gui.SetWindowPos(hwnd, 0, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOZORDER | win32con.SWP_FRAMECHANGED)

    except Exception as e:
        print(f"Error applying focus prevention style: {e}")
        print("Ensure 'pywin32' is installed correctly and running on Windows.")

# --- Graceful Exit Handling ---
def on_closing():
    """Handles cleanup when the Tkinter window is closed (if possible)."""
    print("Close requested. Stopping recording if active...")
    global is_recording, stream
    if is_recording:
        # Set flag first
        is_recording = False
        # Attempt to stop stream quickly
        if stream:
            try:
                if stream.active: stream.stop()
                stream.close()
            except Exception as e: print(f"Error closing stream on exit: {e}")
            finally: stream = None
        # Signal threads (they check is_recording)
        # Give threads a brief moment to finish naturally
        # (Joining here can hang if mainloop is ending)
        # This is best-effort cleanup for overrideredirect windows

    print("Exiting application...")
    root.quit() # Stops mainloop
    root.destroy() # Destroys window


# --- Main Execution ---
if __name__ == "__main__":
    model_loaded = load_vosk_model()

    if model_loaded:
        print("STT Overlay Ready.")
        try:
            device_info = sd.query_devices(DEVICE_ID, 'input')
            print(f"Using device: {device_info['name'] if device_info else 'Default Input Device'}")
            # Check actual sample rate compatibility
            # sd.check_input_settings(device=DEVICE_ID, samplerate=SAMPLE_RATE)
        except Exception as e:
            print(f"Could not query/check audio device: {e}")
            # Consider disabling recording if device query fails critically
            # model_loaded = False # Example: Force disable
    else:
        print("STT Overlay disabled as model failed to load.")

    # --- Crucial: Apply focus prevention *after* window exists but *before* mainloop ---
    root.update_idletasks() # Ensure window elements and HWND are ready
    prevent_focus_steal(root)
    # --- End focus prevention ---

    # Set initial button state based on whether model loaded
    update_button_state()

    # Optional: Add a way to close the overrideredirect window (e.g., keybind)
    # root.bind('<Escape>', lambda e: on_closing()) # Example: Close on Escape key

    # Register the closing handler (might not always work with overrideredirect)
    root.protocol("WM_DELETE_WINDOW", on_closing)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received.")
        on_closing()
    finally:
        # Final cleanup attempt (may not run if process is killed)
        print("Performing final cleanup...")
        if stream and stream.active:
             try:
                 stream.stop()
                 stream.close()
                 print("Final stream cleanup successful.")
             except Exception as e: print(f"Error during final stream cleanup: {e}")