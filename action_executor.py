# action_executor.py
import pyautogui
import subprocess
import time
import sys
import os
import json
import uuid
import tempfile
from datetime import datetime
from PIL import Image
import base64
import io
import requests
import re

# Import the element detector
try:
    from element_detector import ElementDetector, find_ui_elements, get_element_coordinates, describe_element_location
    ELEMENT_DETECTOR_AVAILABLE = True
except ImportError:
    print("Warning: ElementDetector module not available. Advanced UI element detection will be disabled.")
    ELEMENT_DETECTOR_AVAILABLE = False

# --- Configuration & Safety ---
# Disable pyautogui failsafe (moving mouse to corner to stop) - Use with caution!
pyautogui.FAILSAFE = False  # Disabled for Discord automation

# Default wait time after actions like opening apps
DEFAULT_WAIT = 1.5

# Add screen resolution constants
DEFAULT_SCREEN_WIDTH = 1920  # Default target resolution width
DEFAULT_SCREEN_HEIGHT = 1080  # Default target resolution height

# Screenshot directory
SCREENSHOT_DIR = os.path.join(tempfile.gettempdir(), "cua_screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Track the latest screenshot for vision-guided actions
LATEST_SCREENSHOT = {
    "path": None,
    "timestamp": None,
    "width": None,
    "height": None
}

# --- Vision LLM Settings ---
# You can replace this with your own API (OpenAI, etc.) for image analysis
VISION_API_KEY = os.getenv("VISION_API_KEY", os.getenv("OPENMIND_API_KEY"))
VISION_API_URL = os.getenv("VISION_API_URL", "https://api.openmind.org/api/core/gemini/chat/completions")

# --- Action Functions ---

def open_application_via_search(app_name):
    """
    A reliable way to open applications by using the Windows search
    
    Args:
        app_name: Name of the application to open (e.g., 'discord', 'chrome')
        
    Returns:
        True if the action sequence completed, False on error
    """
    try:
        print(f"ACTION: Opening {app_name} via Windows search")
        
        # Open Windows search
        pyautogui.press('win')
        time.sleep(1)
        
        # Type the application name
        pyautogui.write(app_name)
        time.sleep(1)
        
        # Press Enter to launch
        pyautogui.press('enter')
        time.sleep(3)  # Wait for app to launch
        
        print(f"Successfully initiated launch sequence for {app_name}")
        return True
    except Exception as e:
        print(f"Error opening {app_name}: {e}")
        return False

def run_application(app_name):
    """DEPRECATED - Use open_application_via_search instead.
    
    This old function attempts to run an application directly, which may not work
    with modern security settings. It is kept here for backward compatibility only.
    """
    print(f"WARNING: Direct app launching with run_application is deprecated.")
    print(f"Using Windows Search method instead.")
    return open_application_via_search(app_name)

def type_text(text_to_type, interval=0.05):
    """Types the given text using the keyboard."""
    print(f"ACTION: Typing text: '{text_to_type[:50]}...'") # Log snippet
    try:
        # Add a small delay before typing starts, helps ensure focus
        time.sleep(0.3)
        pyautogui.write(text_to_type, interval=interval)
        return True
    except Exception as e:
        print(f"Error typing text: {e}")
        return False

def press_key(key_name):
    """Presses a special key (like Enter, Tab, Ctrl, etc.)."""
    # Add more keys as needed: https://pyautogui.readthedocs.io/en/latest/keyboard.html#keyboard-keys
    valid_keys = [
        'enter', 'tab', 'esc', 'space', 'up', 'down', 'left', 'right',
        'ctrl', 'alt', 'shift', 'win', 'cmd', # Modifier keys
        'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
        'delete', 'backspace', 'home', 'end', 'pageup', 'pagedown', 'insert',
        'printscreen', 'scrolllock', 'pause'
        # Add ctrlleft, ctrlright, shiftleft, etc. if needed
    ]
    key_lower = key_name.lower()
    if key_lower in valid_keys:
        print(f"ACTION: Pressing key: '{key_lower}'")
        try:
            pyautogui.press(key_lower)
            time.sleep(0.3) # Small pause after key press
            return True
        except Exception as e:
            print(f"Error pressing key {key_lower}: {e}")
            return False
    else:
        print(f"Warning: Invalid or unsupported key requested: '{key_name}'")
        return False

def wait(seconds):
    """Pauses execution for a number of seconds."""
    print(f"ACTION: Waiting for {seconds} seconds...")
    try:
        time.sleep(float(seconds))
        return True
    except ValueError:
        print(f"Error: Invalid wait time '{seconds}'")
        return False

def normalize_coordinates(x, y):
    """
    Normalizes coordinates to match the actual screen resolution.
    Takes coordinates that might be based on default 1920x1080 resolution
    and adjusts them to the actual screen resolution.
    
    Args:
        x: X coordinate (potentially based on 1920x1080)
        y: Y coordinate (potentially based on 1920x1080)
        
    Returns:
        Tuple (x, y) with normalized coordinates for actual screen
    """
    try:
        # Convert to integers if they are not already
        x, y = int(float(x)), int(float(y))
        
        # Get actual screen size
        actual_width, actual_height = pyautogui.size()
        
        # If we're already using the target resolution, no need to adjust
        if actual_width == DEFAULT_SCREEN_WIDTH and actual_height == DEFAULT_SCREEN_HEIGHT:
            return x, y
            
        # Calculate scale factors for width and height
        width_scale = actual_width / DEFAULT_SCREEN_WIDTH
        height_scale = actual_height / DEFAULT_SCREEN_HEIGHT
        
        # Scale coordinates
        scaled_x = int(x * width_scale)
        scaled_y = int(y * height_scale)
        
        # Ensure coordinates are within screen bounds
        scaled_x = max(0, min(scaled_x, actual_width - 1))
        scaled_y = max(0, min(scaled_y, actual_height - 1))
        
        # Debug information
        if scaled_x != x or scaled_y != y:
            print(f"Normalized coordinates: ({x}, {y}) -> ({scaled_x}, {scaled_y})")
            print(f"Screen resolution: {actual_width}x{actual_height}, Scale factors: {width_scale:.2f}x{height_scale:.2f}")
        
        return scaled_x, scaled_y
    except Exception as e:
        print(f"Error normalizing coordinates: {e}")
        # Return original coordinates if something went wrong
        return x, y

def click_location(x, y):
    """Moves the mouse to (x, y) and clicks."""
    print(f"ACTION: Clicking at ({x}, {y})")
    try:
        # Convert to integers if they are not already
        x, y = int(float(x)), int(float(y))
        
        # Normalize coordinates for actual screen resolution
        x, y = normalize_coordinates(x, y)
        
        
        # Safety check to avoid clicking at (0,0) which triggers PyAutoGUI failsafe
        # Disable this check when using pre-defined coordinates for Discord
        if (x == 0 and y == 0) and not (192 <= x <= 384 and y >= 1000):  # Allow Discord taskbar area
            print("Warning: Avoiding click at (0,0) which can trigger failsafe")
            # Get the screen size and use a central point instead
            screen_width, screen_height = pyautogui.size()
            # Get a safer default position (center of screen or taskbar)
            x, y = screen_width // 2, screen_height - 40
            print(f"Using safer default position instead: ({x}, {y})")
        
        # Optional bounds checking
        screen_width, screen_height = pyautogui.size()
        if not (0 <= x < screen_width and 0 <= y < screen_height):
            print(f"Warning: Click coordinates ({x},{y}) may be out of screen bounds ({screen_width}x{screen_height}).")
        
        pyautogui.click(x, y)
        time.sleep(0.3)
        return True
    except Exception as e:
        print(f"Error clicking at ({x}, {y}): {e}")
        return False

def execute_comment(comment_text):
    """Prints a comment from the plan (useful for non-executable steps)."""
    print(f"COMMENT: {comment_text}")
    return True

# --- New Screenshot and Vision Functions ---

def take_screenshot(filename=None, region=None):
    """
    Takes a screenshot and saves it to the screenshot directory.
    
    Args:
        filename: Optional custom filename (without extension)
        region: Optional tuple (left, top, width, height) to capture specific region
        
    Returns:
        Path to the saved screenshot
    """
    global LATEST_SCREENSHOT
    
    try:
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}_{uuid.uuid4().hex[:6]}"
            
        # Ensure filename has .png extension
        if not filename.lower().endswith('.png'):
            filename += '.png'
            
        # Create full path
        filepath = os.path.join(SCREENSHOT_DIR, filename)
        
        # Take the screenshot
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()
        
        # Save the screenshot
        screenshot.save(filepath)
        
        # Update latest screenshot info
        LATEST_SCREENSHOT = {
            "path": filepath,
            "timestamp": datetime.now().isoformat(),
            "width": screenshot.width,
            "height": screenshot.height
        }
        
        print(f"ACTION: Screenshot taken and saved to {filepath}")
        return filepath
    
    except Exception as e:
        print(f"Error taking screenshot: {e}")
        return None

def move_mouse_to(x, y, duration=0.5):
    """
    Moves the mouse to the specified coordinates without clicking.
    
    Args:
        x: X coordinate
        y: Y coordinate
        duration: Time (in seconds) the movement should take
    
    Returns:
        True if successful, False otherwise
    """
    try:
        x, y = int(float(x)), int(float(y))
        
        # Normalize coordinates for actual screen resolution
        x, y = normalize_coordinates(x, y)
        
        # Safety check to avoid moving to (0,0) which triggers PyAutoGUI failsafe
        # Disable this check when using pre-defined coordinates for Discord
        if (x == 0 and y == 0) and not (192 <= x <= 384 and y >= 1000):  # Allow Discord taskbar area 
            print("Warning: Avoiding move to (0,0) which can trigger failsafe")
            # Get the screen size and use a central point instead
            screen_width, screen_height = pyautogui.size()
            # Get a safer default position (center of screen or taskbar)
            x, y = screen_width // 2, screen_height - 40
            print(f"Using safer default position instead: ({x}, {y})")
            
        print(f"ACTION: Moving mouse to ({x}, {y})")
        pyautogui.moveTo(x, y, duration=duration)
        return True
    except Exception as e:
        print(f"Error moving mouse to ({x}, {y}): {e}")
        return False

def drag_mouse(start_x, start_y, end_x, end_y, duration=0.5):
    """
    Clicks and drags the mouse from one position to another.
    
    Args:
        start_x, start_y: Starting coordinates
        end_x, end_y: Ending coordinates
        duration: Time (in seconds) the drag should take
    
    Returns:
        True if successful, False otherwise
    """
    try:
        start_x, start_y = int(float(start_x)), int(float(start_y))
        end_x, end_y = int(float(end_x)), int(float(end_y))
        
        # Normalize coordinates for actual screen resolution
        start_x, start_y = normalize_coordinates(start_x, start_y)
        end_x, end_y = normalize_coordinates(end_x, end_y)
        
        print(f"ACTION: Dragging mouse from ({start_x}, {start_y}) to ({end_x}, {end_y})")
        
        # Move to start position
        pyautogui.moveTo(start_x, start_y, duration=duration/2)
        time.sleep(0.1)
        
        # Drag to end position
        pyautogui.dragTo(end_x, end_y, duration=duration, button='left')
        
        return True
    except Exception as e:
        print(f"Error during mouse drag: {e}")
        return False

def right_click(x, y):
    """
    Right-clicks at the specified coordinates.
    
    Args:
        x: X coordinate
        y: Y coordinate
    
    Returns:
        True if successful, False otherwise
    """
    try:
        x, y = int(float(x)), int(float(y))
        
        # Normalize coordinates for actual screen resolution
        x, y = normalize_coordinates(x, y)
        
        # Safety check to avoid clicking at (0,0) which triggers PyAutoGUI failsafe
        # Disable this check when using pre-defined coordinates for Discord
        if (x == 0 and y == 0) and not (192 <= x <= 384 and y >= 1000):  # Allow Discord taskbar area
            print("Warning: Avoiding right-click at (0,0) which can trigger failsafe")
            # Get the screen size and use a central point instead
            screen_width, screen_height = pyautogui.size()
            # Get a safer default position (center of screen or taskbar)
            x, y = screen_width // 2, screen_height - 40
            print(f"Using safer default position instead: ({x}, {y})")
            
        print(f"ACTION: Right-clicking at ({x}, {y})")
        pyautogui.rightClick(x, y)
        time.sleep(0.3)
        return True
    except Exception as e:
        print(f"Error right-clicking at ({x}, {y}): {e}")
        return False

def double_click(x, y):
    """
    Double-clicks at the specified coordinates.
    
    Args:
        x: X coordinate
        y: Y coordinate
    
    Returns:
        True if successful, False otherwise
    """
    try:
        x, y = int(float(x)), int(float(y))
        
        # Normalize coordinates for actual screen resolution
        x, y = normalize_coordinates(x, y)
        
        # Safety check to avoid clicking at (0,0) which triggers PyAutoGUI failsafe
        # Disable this check when using pre-defined coordinates for Discord
        if (x == 0 and y == 0) and not (192 <= x <= 384 and y >= 1000):  # Allow Discord taskbar area
            print("Warning: Avoiding double-click at (0,0) which can trigger failsafe")
            # Get the screen size and use a central point instead
            screen_width, screen_height = pyautogui.size()
            # Get a safer default position (center of screen or taskbar)
            x, y = screen_width // 2, screen_height - 40
            print(f"Using safer default position instead: ({x}, {y})")
            
        print(f"ACTION: Double-clicking at ({x}, {y})")
        pyautogui.doubleClick(x, y)
        time.sleep(0.3)
        return True
    except Exception as e:
        print(f"Error double-clicking at ({x}, {y}): {e}")
        return False

def analyze_screenshot(description=None):
    """
    Analyzes the latest screenshot using a Vision LLM and returns the description.
    
    Args:
        description: Optional instruction for the LLM about what to look for
                    (e.g., "Find the login button")
    
    Returns:
        Analysis results as text
    """
    global LATEST_SCREENSHOT
    
    if not LATEST_SCREENSHOT["path"] or not os.path.exists(LATEST_SCREENSHOT["path"]):
        print("Error: No screenshot available for analysis")
        return "No screenshot available for analysis"
    
    # Use the ElementDetector if available for enhanced UI detection
    element_detection_results = None
    if ELEMENT_DETECTOR_AVAILABLE:
        try:
            print("Using ElementDetector for enhanced UI element detection...")
            detector = ElementDetector()
            
            # Detect UI elements in the screenshot
            results = detector.detect_all(LATEST_SCREENSHOT["path"])
            
            if results["success"]:
                element_count = results["count"]["total"]
                print(f"ElementDetector found {element_count} UI elements")
                
                # Generate a summary of detected elements
                element_detection_results = "\n--- DETECTED UI ELEMENTS ---\n"
                
                # Add text elements with coordinates
                if results["elements"]["text"]:
                    element_detection_results += f"\nTEXT ELEMENTS ({len(results['elements']['text'])}):\n"
                    for i, elem in enumerate(results["elements"]["text"]):
                        coords = get_element_coordinates(elem)
                        if coords:
                            x, y = coords
                            element_detection_results += f"- '{elem['text']}' at ({x}, {y})\n"
                
                # Add button elements with coordinates
                if results["elements"]["buttons"]:
                    element_detection_results += f"\nBUTTON-LIKE ELEMENTS ({len(results['elements']['buttons'])}):\n"
                    for i, elem in enumerate(results["elements"]["buttons"]):
                        coords = get_element_coordinates(elem)
                        if coords:
                            x, y = coords
                            element_detection_results += f"- Button at ({x}, {y})"
                            if elem.get('width') and elem.get('height'):
                                element_detection_results += f", size: {elem['width']}x{elem['height']}"
                            element_detection_results += "\n"
                
                # Generate debug image with highlighted elements
                if os.getenv('CUA_DEBUG', 'False').lower() == 'true':
                    detector.highlight_elements(output_path=os.path.join(SCREENSHOT_DIR, "highlighted_elements.png"))
        except Exception as e:
            print(f"Error using ElementDetector: {e}")
            element_detection_results = None
    
    try:
        # Open and encode the image
        with open(LATEST_SCREENSHOT["path"], "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Get actual screen size for resolution guidance
        actual_width, actual_height = pyautogui.size()
        using_resolution_scaling = actual_width != DEFAULT_SCREEN_WIDTH or actual_height != DEFAULT_SCREEN_HEIGHT
        
        # Prepare system prompt for the vision model
        if not description:
            system_prompt = (
                "You are a computer vision analysis expert that helps with automation. "
                "Analyze this screenshot precisely and thoroughly. "
                "\n\nFocus on: "
                "\n1. Interactive UI elements (buttons, menus, links, input fields, checkboxes, etc.)"
                "\n2. Text content that would be useful for navigation"
                "\n3. Notable sections or areas of the interface"
                "\n\nFor EVERY interactive element, provide:"
                "\n- Element type (button, link, menu item, etc.)"
                "\n- Text content or label (if any)"
                "\n- PRECISE pixel coordinates (x,y) of its center point"
                "\n- Description of its appearance"
                "\n- Relative position on screen"
                "\n\nThe image dimensions are: "
                f"{LATEST_SCREENSHOT['width']}x{LATEST_SCREENSHOT['height']}."
                f"\nThe target system screen resolution is: {DEFAULT_SCREEN_WIDTH}x{DEFAULT_SCREEN_HEIGHT}."
            )
            
            # Add information about coordinate normalization if needed
            if using_resolution_scaling:
                width_scale = actual_width / DEFAULT_SCREEN_WIDTH
                height_scale = actual_height / DEFAULT_SCREEN_HEIGHT
                
                system_prompt += (
                    f"\n\nIMPORTANT - COORDINATE SCALING:"
                    f"\nThe system will automatically normalize coordinates. For best accuracy:"
                    f"\n- Provide coordinates based on a {DEFAULT_SCREEN_WIDTH}x{DEFAULT_SCREEN_HEIGHT} resolution"
                    f"\n- Coordinates will be automatically scaled by factors: {width_scale:.2f}x, {height_scale:.2f}y"
                    f"\n- This means your coordinates will be transformed to the actual screen resolution of {actual_width}x{actual_height}"
                )
            
            system_prompt += (
                "\n\nFormat your response in a clean, structured way. Group similar elements together."
                "\nBe extremely precise with coordinates - they will be used for exact mouse positioning."
            )
        else:
            system_prompt = (
                f"You are a computer vision analysis expert that helps with automation. "
                f"Analyze this screenshot with a focus on: {description}"
                "\n\nFor ALL elements matching this criteria, provide:"
                "\n- Element type (button, link, menu item, etc.)"
                "\n- Text content or label (if any)"
                "\n- PRECISE pixel coordinates (x,y) of its center point"
                "\n- Description of its appearance"
                "\n- Relative position on screen"
                "\n\nThe image dimensions are: "
                f"{LATEST_SCREENSHOT['width']}x{LATEST_SCREENSHOT['height']}."
                f"\nThe target system screen resolution is: {DEFAULT_SCREEN_WIDTH}x{DEFAULT_SCREEN_HEIGHT}."
            )
            
            # Add information about coordinate normalization if needed
            if using_resolution_scaling:
                width_scale = actual_width / DEFAULT_SCREEN_WIDTH
                height_scale = actual_height / DEFAULT_SCREEN_HEIGHT
                
                system_prompt += (
                    f"\n\nIMPORTANT - COORDINATE SCALING:"
                    f"\nThe system will automatically normalize coordinates. For best accuracy:"
                    f"\n- Provide coordinates based on a {DEFAULT_SCREEN_WIDTH}x{DEFAULT_SCREEN_HEIGHT} resolution"
                    f"\n- Coordinates will be automatically scaled by factors: {width_scale:.2f}x, {height_scale:.2f}y"
                    f"\n- This means your coordinates will be transformed to the actual screen resolution of {actual_width}x{actual_height}"
                )
            
            system_prompt += (
                "\n\nFormat coordinates as exact numbers that can be used for mouse positioning."
                "\nExample: 'File menu: located at coordinates (45, 23)'"
                "\n\nBe extremely precise with coordinates - they will be used for exact mouse positioning."
            )
        
        # Add element detection results to the prompt if available
        if element_detection_results:
            system_prompt += "\n\nThe following UI elements were automatically detected using computer vision techniques. USE THESE PRECISE COORDINATES when referring to these elements:\n"
            system_prompt += element_detection_results
            
        # Prepare API request payload
        headers = {
            "x-api-key": VISION_API_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gemini-2.0-flash-exp", # Using a vision-capable model (change as needed)
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Analyze this screenshot and identify all UI elements with their precise locations and descriptions."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.3  # Lower temperature for more precise, consistent responses
        }
        
        print(f"Analyzing screenshot via Vision API...")
        response = requests.post(VISION_API_URL, headers=headers, json=payload, timeout=120) # Increased timeout for vision processing
        response.raise_for_status()
        
        result = response.json()
        
        # Extract analysis text
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            analysis = message.get("content")
            if analysis:
                print("Screenshot analysis complete.")
                
                # Add a reference to the automatic element detection if used
                if element_detection_results:
                    analysis = "ENHANCED ANALYSIS (with computer vision element detection):\n\n" + analysis
                
                # Add a reminder about coordinate normalization if applicable
                if using_resolution_scaling:
                    analysis = (
                        f"COORDINATE SCALING ACTIVE: Coordinates will be automatically scaled from {DEFAULT_SCREEN_WIDTH}x{DEFAULT_SCREEN_HEIGHT} "
                        f"to your actual screen resolution of {actual_width}x{actual_height}.\n\n" + analysis
                    )
                
                # Store the analysis in LATEST_SCREENSHOT for reference by element finders
                LATEST_SCREENSHOT["analysis"] = analysis
                
                return analysis.strip()
            else:
                return "Error: Vision API returned empty analysis"
        else:
            return "Error: Could not parse Vision API response"
            
    except Exception as e:
        print(f"Error analyzing screenshot: {e}")
        return f"Error analyzing screenshot: {str(e)}"

def locate_on_screen(image_to_find, confidence=0.8):
    """
    Locates an image on the screen.
    
    Args:
        image_to_find: Path to the image file to find
        confidence: Confidence level (0-1) for matching
    
    Returns:
        Tuple with (x, y) coordinates of the center of the found image, or None if not found
    """
    try:
        location = pyautogui.locateOnScreen(image_to_find, confidence=confidence)
        if location:
            # Get center point of the located image
            center = pyautogui.center(location)
            print(f"Found image at position {center}")
            return center
        else:
            print(f"Image not found on screen: {image_to_find}")
            return None
    except Exception as e:
        print(f"Error locating image on screen: {e}")
        return None
        
def scroll(direction, clicks=3):
    """
    Scrolls the page in specified direction.
    
    Args:
        direction: 'up' or 'down'
        clicks: Number of scroll units (positive integer)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        amount = abs(int(clicks))
        if direction.lower() == 'down':
            print(f"ACTION: Scrolling down {amount} clicks")
            pyautogui.scroll(-amount)  # Negative value scrolls down
        elif direction.lower() == 'up':
            print(f"ACTION: Scrolling up {amount} clicks")
            pyautogui.scroll(amount)   # Positive value scrolls up
        else:
            print(f"Error: Invalid scroll direction '{direction}'")
            return False
            
        time.sleep(0.5)
        return True
    except Exception as e:
        print(f"Error scrolling {direction}: {e}")
        return False

# --- Plan Parsers ---

def execute_plan_step(action_obj, use_json_format=True):
    """Execute a single action object or command and return success status."""
    if use_json_format:
        # JSON format
        if not isinstance(action_obj, dict) or "action" not in action_obj:
            print(f"Skipping invalid action object: {action_obj}")
            return False
        
        command_name = action_obj["action"]
        try:
            # Original actions
            if command_name == "run_app":
                # This is now deprecated - we should warn the user
                app = action_obj.get("app_name")
                print(f"WARNING: Direct app launching with 'run_app' is deprecated. The agent should use vision-guided actions to find and open {app}.")
                print(f"Attempting to press the Windows key to open the start menu instead.")
                press_key("win")
                time.sleep(1)
                return True  # We'll handle this differently now
            elif command_name == "type_text":
                text = action_obj.get("text")
                if text is not None: 
                    return type_text(text)
                else: 
                    print("Error: 'text' missing for type_text")
            elif command_name == "press_key":
                key = action_obj.get("key")
                if key: 
                    return press_key(key)
                else: 
                    print("Error: 'key' missing for press_key")
            elif command_name == "wait":
                sec = action_obj.get("seconds")
                if sec is not None: 
                    return wait(sec)
                else: 
                    print("Error: 'seconds' missing for wait")
            elif command_name == "click":
                x = action_obj.get("x")
                y = action_obj.get("y")
                if x is not None and y is not None: 
                    return click_location(x, y)
                else: 
                    print("Error: 'x' or 'y' missing for click")
            elif command_name == "comment":
                comment = action_obj.get("comment_text")
                if comment: 
                    return execute_comment(comment)
                else: 
                    print("Error: 'comment_text' missing for comment")
                
            # Vision-based actions
            elif command_name == "screenshot":
                filename = action_obj.get("filename")
                region = action_obj.get("region")
                
                if region and isinstance(region, list) and len(region) == 4:
                    return take_screenshot(filename, tuple(region)) is not None
                else:
                    return take_screenshot(filename) is not None
                    
            elif command_name == "analyze_screen":
                description = action_obj.get("description")
                analysis = analyze_screenshot(description)
                print(f"SCREEN ANALYSIS:\n{analysis}")
                return True
                
            elif command_name == "move_mouse":
                x = action_obj.get("x")
                y = action_obj.get("y")
                duration = action_obj.get("duration", 0.5)
                
                if x is not None and y is not None:
                    return move_mouse_to(x, y, duration)
                else:
                    print("Error: 'x' or 'y' missing for move_mouse")
                    
            elif command_name == "right_click":
                x = action_obj.get("x")
                y = action_obj.get("y")
                
                if x is not None and y is not None:
                    return right_click(x, y)
                else:
                    print("Error: 'x' or 'y' missing for right_click")
                    
            elif command_name == "double_click":
                x = action_obj.get("x")
                y = action_obj.get("y")
                
                if x is not None and y is not None:
                    return double_click(x, y)
                else:
                    print("Error: 'x' or 'y' missing for double_click")
                    
            elif command_name == "drag":
                start_x = action_obj.get("start_x")
                start_y = action_obj.get("start_y")
                end_x = action_obj.get("end_x")
                end_y = action_obj.get("end_y")
                duration = action_obj.get("duration", 0.5)
                
                if all(v is not None for v in [start_x, start_y, end_x, end_y]):
                    return drag_mouse(start_x, start_y, end_x, end_y, duration)
                else:
                    print("Error: Missing coordinates for drag")
                    
            elif command_name == "scroll":
                direction = action_obj.get("direction")
                amount = action_obj.get("amount", 3)
                
                if direction:
                    return scroll(direction, amount)
                else:
                    print("Error: 'direction' missing for scroll")
            else:
                print(f"Warning: Unknown command '{command_name}'")
                return False
        except Exception as e:
            print(f"Critical error executing {action_obj}: {e}")
            return False
    else:
        # Simple list format
        command_part = action_obj
        try:
            # Handle commands with arguments
            if '(' in command_part and command_part.endswith(')'):
                command_name = command_part[:command_part.index('(')]
                args_part = command_part[command_part.index('(')+1 : command_part.rindex(')')]
            else:
                # Assume command with no args if format is wrong
                print(f"Warning: Could not parse arguments from command line: {command_part}. Assuming no args.")
                command_name = command_part
                args_part = "" # No arguments
        except ValueError:
            print(f"Error parsing command line: {command_part}")
            return False

        try:
            # Original actions
            if command_name == "run_app":
                # This is now deprecated - we should warn the user
                print(f"WARNING: Direct app launching with 'run_app' is deprecated. The agent should use vision-guided actions to find and open {args_part}.")
                print(f"Attempting to press the Windows key to open the start menu instead.")
                press_key("win")
                time.sleep(1)
                return True  # We'll handle this differently now
            elif command_name == "type_text":
                # Handle potential quotes
                return type_text(args_part.strip("'\""))
            elif command_name == "press_key":
                return press_key(args_part.strip("'\""))
            elif command_name == "wait":
                return wait(args_part)
            elif command_name == "click":
                # Parse "x, y"
                coords = [c.strip() for c in args_part.split(',')]
                if len(coords) == 2:
                    return click_location(coords[0], coords[1])
                else: 
                    print(f"Error: Invalid args for click: {args_part}")
                    return False
            elif command_name == "comment":
                return execute_comment(args_part.strip("'\""))
                
            # Vision-based actions
            elif command_name == "screenshot":
                # Optional filename
                filename = args_part.strip("'\"") if args_part else None
                return take_screenshot(filename) is not None
            elif command_name == "move_mouse":
                # Parse "x, y"
                coords = [c.strip() for c in args_part.split(',')]
                if len(coords) == 2:
                    return move_mouse_to(coords[0], coords[1])
                else: 
                    print(f"Error: Invalid args for move_mouse: {args_part}")
                    return False
            elif command_name == "analyze_screen":
                # Optional description to look for
                description = args_part.strip("'\"") if args_part else None
                analysis = analyze_screenshot(description)
                print(f"SCREEN ANALYSIS:\n{analysis}")
                return True
            elif command_name == "right_click":
                # Parse "x, y"
                coords = [c.strip() for c in args_part.split(',')]
                if len(coords) == 2:
                    return right_click(coords[0], coords[1])
                else: 
                    print(f"Error: Invalid args for right_click: {args_part}")
                    return False
            elif command_name == "double_click":
                # Parse "x, y"
                coords = [c.strip() for c in args_part.split(',')]
                if len(coords) == 2:
                    return double_click(coords[0], coords[1])
                else: 
                    print(f"Error: Invalid args for double_click: {args_part}")
                    return False
            elif command_name == "drag":
                # Parse "start_x, start_y, end_x, end_y"
                coords = [c.strip() for c in args_part.split(',')]
                if len(coords) == 4:
                    return drag_mouse(coords[0], coords[1], coords[2], coords[3])
                else: 
                    print(f"Error: Invalid args for drag: {args_part}")
                    return False
            elif command_name == "scroll":
                # Parse "direction, amount"
                args = [a.strip() for a in args_part.split(',')]
                if len(args) >= 1:
                    direction = args[0].strip("'\"")
                    amount = int(args[1]) if len(args) > 1 else 3
                    return scroll(direction, amount)
                else: 
                    print(f"Error: Invalid args for scroll: {args_part}")
                    return False
            else:
                print(f"Warning: Unknown command '{command_name}'")
                return False
        except Exception as e:
            print(f"Critical error executing {command_part}: {e}")
            return False
    
    return False  # Default return if no action was executed

def execute_plan_simple_list(plan_text, vision_api_callback=None):
    """
    Executes a plan formatted as 'EXECUTE: command(args)'.
    
    Args:
        plan_text: The plan text
        vision_api_callback: Function to call after screenshots to get new actions
    """
    print("\n--- Executing Plan (Simple List) ---")
    lines = plan_text.strip().split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        
        if not line.startswith("EXECUTE:"):
            if line: 
                print(f"Skipping non-execute line: {line}")
            continue

        command_part = line[len("EXECUTE:"):].strip()
        
        # Execute the current step
        success = execute_plan_step(command_part, use_json_format=False)
        
        if not success:
            print(f"Execution failed for: {line}. Stopping plan.")
            break
        
        # Check if we just took a screenshot and need vision-guided actions
        if "screenshot" in command_part.lower() and vision_api_callback:
            # Check if the next command is analyze_screen
            analyze_next = False
            if i < len(lines) and "analyze_screen" in lines[i].lower():
                # Let the analyze_screen command run first
                analyze_next = True
            
            if not analyze_next or i >= len(lines) - 1:
                # We just took a screenshot, but there's no analyze_screen command coming,
                # or we've analyzed but this is the end of the plan
                # Let's ask the LLM what to do next based on the screenshot
                print("\n--- Getting vision-guided actions based on the screenshot ---")
                
                # Call back to the main script to get new actions based on the screenshot
                new_plan = vision_api_callback(LATEST_SCREENSHOT)
                
                if new_plan and new_plan.strip():
                    # We got a new plan! Insert these new commands into our execution queue
                    new_lines = new_plan.strip().split('\n')
                    # Insert the new lines into our execution queue, right after the current position
                    lines[i:i] = new_lines
                    print(f"Added {len(new_lines)} new vision-guided actions to the plan")
    
    print("--- Plan Execution Finished ---")

def execute_plan_json(plan_json_string, vision_api_callback=None):
    """
    Executes a plan formatted as a JSON list of action objects.
    
    Args:
        plan_json_string: JSON string containing the action plan
        vision_api_callback: Function to call after screenshots to get new actions
    """
    print("\n--- Executing Plan (JSON) ---")
    try:
        # Try to handle potential markdown code block fences
        if plan_json_string.strip().startswith("```json"):
            plan_json_string = plan_json_string.strip()[7:]
        if plan_json_string.strip().endswith("```"):
            plan_json_string = plan_json_string.strip()[:-3]

        actions = json.loads(plan_json_string.strip())
        if not isinstance(actions, list):
            print("Error: Plan is not a JSON list.")
            return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON plan: {e}")
        print(f"Received:\n{plan_json_string}")
        return
    
    i = 0
    while i < len(actions):
        action_obj = actions[i]
        i += 1
        
        if not isinstance(action_obj, dict) or "action" not in action_obj:
            print(f"Skipping invalid action object: {action_obj}")
            continue
        
        # Special handling for find_element action
        if action_obj.get("action") == "find_element" and ELEMENT_DETECTOR_AVAILABLE:
            description = action_obj.get("description")
            element_type = action_obj.get("type")
            
            if description:
                print(f"ACTION: Finding element matching '{description}'")
                element = find_element_by_description(description, element_type)
                
                if element and get_element_coordinates(element):
                    # If the next action is a mouse action, update its coordinates
                    if i < len(actions) and actions[i].get("action") in ["move_mouse", "click", "right_click", "double_click"]:
                        x, y = get_element_coordinates(element)
                        next_action = actions[i]
                        print(f"Updating coordinates for {next_action['action']} to ({x}, {y})")
                        next_action["x"] = x
                        next_action["y"] = y
                        
                    # Log the finding
                    location_desc = describe_element_location(element)
                    print(f"Element found: {location_desc}")
                    continue
                else:
                    print(f"Warning: Could not find element matching '{description}'")
                    continue
        
        # Execute the current step
        success = execute_plan_step(action_obj, use_json_format=True)
        
        if not success:
            print(f"Execution failed for: {action_obj}. Stopping plan.")
            break
        
        # Check if we just took a screenshot and should get vision-guided actions
        if action_obj.get("action") == "screenshot" and vision_api_callback:
            # Check if the next action is analyze_screen
            analyze_next = False
            if i < len(actions) and actions[i].get("action") == "analyze_screen":
                # Let the analyze_screen command run first
                analyze_next = True
            
            if not analyze_next or i >= len(actions) - 1:
                # We just took a screenshot, but there's no analyze_screen command coming,
                # or we've analyzed but this is the end of the plan
                # Let's ask the LLM what to do next based on the screenshot
                print("\n--- Getting vision-guided actions based on the screenshot ---")
                
                # Call back to the main script to get new actions based on the screenshot
                new_plan_json = vision_api_callback(LATEST_SCREENSHOT)
                
                if new_plan_json and new_plan_json.strip():
                    try:
                        # Clean up the new plan string if needed
                        if new_plan_json.strip().startswith("```json"):
                            new_plan_json = new_plan_json.strip()[7:]
                        if new_plan_json.strip().endswith("```"):
                            new_plan_json = new_plan_json.strip()[:-3]
                            
                        # Parse the new actions
                        new_actions = json.loads(new_plan_json.strip())
                        if isinstance(new_actions, list):
                            # Insert the new actions into our execution queue, right after the current position
                            actions[i:i] = new_actions
                            print(f"Added {len(new_actions)} new vision-guided actions to the plan")
                        else:
                            print("Warning: New vision-guided plan is not a JSON list, skipping")
                    except json.JSONDecodeError as e:
                        print(f"Error decoding vision-guided JSON plan: {e}")
    
    print("--- Plan Execution Finished ---")

# Add new function to find elements by text or type
def find_element_by_description(description, element_type=None):
    """
    Find UI elements in the latest screenshot that match a description.
    
    Args:
        description: Text description to search for
        element_type: Optional type of element ('text', 'button', etc.)
    
    Returns:
        Dictionary with element information including coordinates, or None if not found
    """
    global LATEST_SCREENSHOT
    
    # Special case for app launches - use the more reliable Windows search method
    if description.lower() == "discord" and (element_type is None or element_type.lower() in ["app", "application"]):
        print("Instead of locating Discord, will launch it directly via Windows search")
        open_application_via_search("discord")
        time.sleep(3)  # Wait for Discord to open
        
        # Return a dummy element to continue the workflow
        screen_width, screen_height = pyautogui.size()
        return {
            'type': 'direct_launch',
            'description': 'Discord (launched via Windows search)',
            'x': screen_width // 2,  # Center of screen 
            'y': screen_height // 2   # Center of screen
        }
    
    # Special case for Discord login
    if "login" in description.lower() and "discord" in description.lower():
        print("Special case: Discord login field")
        screen_width, screen_height = pyautogui.size()
        return {
            'type': 'text_field',
            'description': 'Discord login field',
            'x': screen_width // 2,  # Center of screen 
            'y': screen_height // 2 - 100  # Slightly above center
        }
    
    # Special case for Discord contacts/text fields
    if "geekgoop" in description.lower():
        print("Using special case for Discord contact 'geekgoop'")
        screen_width, screen_height = pyautogui.size()
        element = {
            'type': 'contact',
            'description': 'geekgoop',
            'x': screen_width // 5,    # Left sidebar of Discord
            'y': screen_height // 2     # Middle of screen
        }
        print(f"Using fallback geekgoop position at ({element['x']}, {element['y']})")
        return element
    
    if not LATEST_SCREENSHOT["path"] or not os.path.exists(LATEST_SCREENSHOT["path"]):
        print("Error: No screenshot available for element search")
        return None
        
    # First try to use ElementDetector if available
    if ELEMENT_DETECTOR_AVAILABLE:
        try:
            # Find UI elements in the screenshot
            elements = find_ui_elements(LATEST_SCREENSHOT["path"], element_type, description)
            
            if elements:
                print(f"Found {len(elements)} elements matching '{description}'")
                # Return the first matching element
                element = elements[0]
                coords = get_element_coordinates(element)
                
                if coords:
                    print(f"Selected element at coordinates {coords}")
                    return element
        except Exception as e:
            print(f"Warning: Element detector error: {e}")
            # Continue to fallback method
    
    # Special case for Discord-related searches when vision fails
    if description.lower() == "discord":
        print("Using special case for Discord application")
        # Create a synthetic element for Discord icon (typical position in taskbar)
        screen_width, screen_height = pyautogui.size()
        # Try different positions along the taskbar where Discord might be
        positions = [
            (screen_width // 10, screen_height - 25),    # Left side of taskbar
            (screen_width // 6, screen_height - 25),     # Another possible position
            (screen_width // 4, screen_height - 25),     # Another possible position
            (min(screen_width - 200, 800), screen_height - 25)  # Right side of taskbar (but not too far right)
        ]
        
        # Choose position based on a simple heuristic (adjust taskbar position based on screen width)
        position_index = min(3, max(0, (screen_width // 800) - 1))
        discord_x, discord_y = positions[position_index]
        
        element = {
            'type': 'application',
            'description': 'Discord',
            'x': discord_x,
            'y': discord_y
        }
        print(f"Using fallback Discord position at ({element['x']}, {element['y']})")
        return element
    
    if "search" in description.lower():
        print("Using special case for Discord search bar")
        screen_width, screen_height = pyautogui.size()
        element = {
            'type': 'search',
            'description': 'Search',
            'x': screen_width // 5,     # Left sidebar
            'y': 80                     # Top area of Discord
        }
        print(f"Using fallback search position at ({element['x']}, {element['y']})")
        return element
    
    if "message input" in description.lower():
        print("Using special case for Discord message input field")
        screen_width, screen_height = pyautogui.size()
        element = {
            'type': 'input',
            'description': 'Message Input',
            'x': screen_width // 2,     # Center width
            'y': screen_height - 100    # Near bottom of screen
        }
        print(f"Using fallback message input position at ({element['x']}, {element['y']})")
        return element
    
    # Fallback: Try to extract coordinates from the latest screenshot analysis
    if hasattr(LATEST_SCREENSHOT, "analysis") and LATEST_SCREENSHOT.get("analysis"):
        analysis = LATEST_SCREENSHOT["analysis"]
        lines = analysis.split("\n")
        
        # Search for the description and coordinates in the analysis text
        for i, line in enumerate(lines):
            # Look for the description in the line
            if description.lower() in line.lower():
                # Look for coordinates in this line or the next few lines
                for j in range(i, min(i+5, len(lines))):
                    coord_match = re.search(r'\((\d+),\s*(\d+)\)', lines[j])
                    if coord_match:
                        x, y = int(coord_match.group(1)), int(coord_match.group(2))
                        print(f"Fallback: Found coordinates ({x}, {y}) for '{description}' in vision analysis")
                        
                        # Create a synthetic element
                        element = {
                            'type': element_type or 'vision_element',
                            'description': description,
                            'x': x,
                            'y': y
                        }
                        return element
        
        print(f"Could not find coordinates for '{description}' in vision analysis")
    
    # If we get here, we couldn't find the element
    print(f"No elements found matching '{description}'")
    return None