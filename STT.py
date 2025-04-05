import vosk
import sounddevice as sd
import queue
import threading
import json
import pyautogui
import sys
import os
import time
from pathlib import Path


# --- Configuration ---
MODEL_PATH = "/Users/amirkhabaza/Quick/CUA_Project/vosk-model-en-us-0.42-gigaspeech"
SAMPLE_RATE = 16000
DEVICE_ID = 0
BLOCK_SIZE = 8000

# --- Global Variables ---
q = queue.Queue()
result_queue = queue.Queue()
is_recording = False
stream = None
recognizer = None
model = None
processing_thread = None
typing_thread = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "vosk-model-en-us-0.42-gigaspeech"

# --- Vosk Model Loading ---


def load_vosk_model():
    global model

    # Ensure path is a string, not a PosixPath object
    from pathlib import Path
    raw_model_path = Path(
        "/Users/amirkhabaza/Downloads/Quick/CUA_Project/vosk-model-en-us-0.42-gigaspeech")
    model_path_str = str(raw_model_path)

    if not os.path.exists(model_path_str):
        print(f"ERROR: Vosk model folder not found at '{model_path_str}'")
        print("Please download a model from https://alphacephei.com/vosk/models")
        return False

    try:
        print(f"Loading Vosk model from: {model_path_str}")
        model = vosk.Model(model_path_str)  # MUST be a string
        print("Vosk model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading Vosk model: {e}")
        return False
# --- Audio Callback ---


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    if is_recording:
        try:
            q.put(bytes(indata))
        except Exception as e:
            print(f"Error putting data into queue: {e}")

# --- Audio Processing Thread ---


def process_audio():
    global recognizer, is_recording
    if recognizer is None:
        print("ERROR: Recognizer not initialized before processing thread start.")
        result_queue.put(None)
        return

    print("Audio processing thread started (using existing recognizer).")
    while is_recording:
        try:
            data = q.get(timeout=0.1)
            if not is_recording or recognizer is None:
                break

            if recognizer.AcceptWaveform(data):
                result_json = recognizer.Result()
                result_dict = json.loads(result_json)
                final_text = result_dict.get('text', '').strip()
                if final_text.lower() == "the":
                    print("Filtered spurious 'the'")
                elif final_text:
                    result_queue.put(("final", final_text))
                    print(f"Final: {final_text}")
            else:
                partial_json = recognizer.PartialResult()
                partial_dict = json.loads(partial_json)
                partial_text = partial_dict.get('partial', '')
                if partial_text:
                    result_queue.put(("partial", partial_text))
        except queue.Empty:
            continue
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            continue
        except Exception as e:
            print(f"Error in audio processing loop: {e}")
            break

    print("Processing final audio chunk...")
    try:
        if recognizer:
            final_result_json = recognizer.FinalResult()
            final_result_dict = json.loads(final_result_json)
            final_text = final_result_dict.get('text', '').strip()
            if final_text.lower() == "the":
                print("Filtered final 'the'")
            elif final_text:
                result_queue.put(("final", final_text))
                print(f"Final (at end): {final_text}")
        else:
            print("Recognizer was None during final processing.")
    except Exception as e:
        print(f"Error getting final result: {e}")

    print("Audio processing thread finished.")
    result_queue.put(None)

# --- Typing Thread ---


def type_results():
    global is_recording
    print("Typing thread started.")
    current_segment_text = ""
    while True:
        try:
            item = result_queue.get()
            if item is None:
                if current_segment_text:
                    pyautogui.write(' ')
                    current_segment_text = ""
                break
            type_flag, text = item
            if not text and type_flag != "final":
                continue
            if type_flag == "partial":
                if text.startswith(current_segment_text):
                    new_chars = text[len(current_segment_text):]
                    if new_chars:
                        pyautogui.write(new_chars, interval=0.01)
                        current_segment_text = text
                else:
                    if len(text) > len(current_segment_text):
                        new_chars = text[len(current_segment_text):]
                        pyautogui.write(new_chars, interval=0.01)
                        current_segment_text = text
            elif type_flag == "final":
                if text.startswith(current_segment_text):
                    new_chars = text[len(current_segment_text):]
                    if new_chars:
                        pyautogui.write(new_chars, interval=0.01)
                elif text:
                    pyautogui.write(text, interval=0.01)
                pyautogui.write(' ', interval=0.01)
                current_segment_text = ""
        except Exception as e:
            print(f"Error in typing thread: {e}")
            break
    print("Typing thread finished.")

# --- Toggle Recording (Command-Line Based) ---


def toggle_recording():
    global is_recording, stream, processing_thread, typing_thread, recognizer, model
    if not is_recording:
        if not model:
            print("ERROR: Vosk model not loaded. Cannot start recording.")
            return
        try:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            while not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    break

            print("Creating new recognizer instance.")
            recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
            recognizer.SetWords(False)
            is_recording = True

            try:
                sd.check_input_settings(
                    device=DEVICE_ID, samplerate=SAMPLE_RATE, channels=1)
            except Exception as e:
                print(f"Error checking input device settings: {e}")
                is_recording = False
                recognizer = None
                return

            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                device=DEVICE_ID,
                channels=1,
                dtype='int16',
                callback=audio_callback,
            )
            stream.start()
            print("Recording started...")

            processing_thread = threading.Thread(
                target=process_audio, daemon=True)
            typing_thread = threading.Thread(target=type_results, daemon=True)
            processing_thread.start()
            typing_thread.start()

        except Exception as e:
            print(f"Error starting recording: {e}")
            is_recording = False
            if stream and stream.active:
                try:
                    stream.stop()
                    stream.close()
                except Exception as e_close:
                    print(
                        f"Error closing stream during start error: {e_close}")
            stream = None
            recognizer = None
    else:
        print("Stopping recording...")
        is_recording = False
        if stream:
            try:
                if stream.active:
                    stream.stop()
                stream.close()
            except Exception as e_close:
                print(f"Error stopping/closing stream: {e_close}")
            finally:
                stream = None
        if processing_thread and processing_thread.is_alive():
            processing_thread.join(timeout=2.0)
        if typing_thread and typing_thread.is_alive():
            typing_thread.join(timeout=1.5)
        print("Cleaning up recognizer instance.")
        recognizer = None

# --- Main Execution ---


def main():
    global model
    if load_vosk_model():
        print("Vosk model loaded.")
    else:
        print("Failed to load Vosk model.")
        return
    print("STT is ready.")
    print("Enter 'r' to toggle recording, or 'q' to quit.")
    while True:
        cmd = input("Command (r = toggle record, q = quit): ").strip().lower()
        if cmd == 'r':
            toggle_recording()
        elif cmd == 'q':
            if is_recording:
                toggle_recording()
            break
        else:
            print("Unknown command. Please enter 'r' or 'q'.")


model = None  # global variabl e


def initialize_model():
    global model
    if load_vosk_model():
        print("vosk model loaded.")
    else:
        print("Failed")


if __name__ == "__main__":
    initialize_model()
    main()
