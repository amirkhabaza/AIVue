# cua_main.py
import requests
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

# --- Import the executor functions ---
# (Make sure action_executor.py is in the same directory)
try:
    from action_executor import execute_plan_simple_list, execute_plan_json, LATEST_SCREENSHOT, analyze_screenshot
except ImportError:
    print("Error: Could not import action_executor.py.")
    print("Please ensure action_executor.py is in the same directory as this script.")
    exit()

# --- Configuration ---

# !! IMPORTANT: Set your API key as an environment variable !!
# On Linux/macOS: export OPENMIND_API_KEY='your_api_key_here'
# On Windows (cmd): set OPENMIND_API_KEY=your_api_key_here
# On Windows (PowerShell): $env:OPENMIND_API_KEY='your_api_key_here'
API_KEY = os.getenv("OPENMIND_API_KEY")

if not API_KEY:
    print("Error: OPENMIND_API_KEY environment variable not set.")
    print("Please set the environment variable with your OpenMind API key.")
    exit()

# !! IMPORTANT: Choose your provider and model !!
# Provider options: "openai", "deepseek", "gemini", "xai"
OPENMIND_PROVIDER = "gemini" # <-- CHANGE AS NEEDED

# Model options depend on the provider (check Openmind docs):
# e.g., for "openai": "gpt-4o", "gpt-4o-mini"
# e.g., for "deepseek": "deepseek-chat"
# e.g., for "gemini": "gemini-2.0-flash-exp"
# e.g., for "xai": "grok-2-latest"
OPENMIND_MODEL_ID = "gemini-2.0-flash-exp" # <-- CHANGE AS NEEDED (must be valid for the chosen provider)

# API Endpoint URL template
OPENMIND_API_URL_TEMPLATE = "https://api.openmind.org/api/core/gemini/chat/completions"

# --- CHOOSE WHICH PLAN FORMAT YOU WANT THE LLM TO GENERATE ---
# Set to True if your system prompt asks for JSON output.
# Set to False if your system prompt asks for the 'EXECUTE: command(args)' list.
USE_JSON_FORMAT = True # <-- IMPORTANT: Match this with your system prompt!
# --- ---

# --- Vision Analysis Toggle ---
# Set to True to enable vision-based analysis of screenshots
# Set to False to disable (if you don't have access to a vision-capable model)
USE_VISION_ANALYSIS = True # <-- IMPORTANT: Set to False if you don't have a vision model API
# --- ---

# --- LLM Interaction Function (Updated) ---
def get_llm_interpretation(user_command, screenshot_data=None, is_vision_follow_up=False):
    """
    Sends the user command to the Openmind LLM API and gets its interpretation/plan.
    
    Args:
        user_command: The user's command text
        screenshot_data: Optional dict with screenshot info if using vision guidance
        is_vision_follow_up: Whether this is a follow-up plan after a screenshot
    """
    api_url = OPENMIND_API_URL_TEMPLATE.format(provider=OPENMIND_PROVIDER)

    headers = {
        "x-api-key": API_KEY, # Use x-api-key header
        "Content-Type": "application/json",
    }

    # Get screen resolution for coordinate guidance
    try:
        import pyautogui
        actual_width, actual_height = pyautogui.size()
        resolution_info = f"The user's actual screen resolution is {actual_width}x{actual_height}. "
        standard_resolution = "However, all coordinates should be provided for a standard 1920x1080 resolution, as they will be automatically adjusted by the system."
    except ImportError:
        resolution_info = ""
        standard_resolution = ""

    # --- Choose the System Prompt based on USE_JSON_FORMAT ---
    if USE_JSON_FORMAT:
        if is_vision_follow_up:
            # Special prompt for vision follow-up actions
            system_prompt_content = (
                "You are a computer vision-guided execution assistant. "
                "Based on the screenshot analysis, create a plan of what to do next. "
                "Respond ONLY with a valid JSON list of action objects. "
                "IMPORTANT: Focus ONLY on mouse movements and interactions based on what you see in the screenshot. "
                "DO NOT suggest taking another screenshot until you've interacted with the elements from the current one. "
                f"\n\n{resolution_info}{standard_resolution}"
                "\n\nFor this follow-up plan, focus on these actions:"
                "\n- 'find_element': Find an element by description. Parameters: 'description', 'type' (optional: 'text', 'button')"
                "\n- 'move_mouse': Move the mouse to an element. Parameters: 'x', 'y'"
                "\n- 'click': Click on an element. Parameters: 'x', 'y'"
                "\n- 'right_click': Right-click at coordinates. Parameters: 'x', 'y'"
                "\n- 'double_click': Double-click at coordinates. Parameters: 'x', 'y'"
                "\n- 'drag': Drag from one position to another. Parameters: 'start_x', 'start_y', 'end_x', 'end_y'"
                "\n- 'type_text': Type text if needed. Parameters: 'text'"
                "\n- 'press_key': Press keys if needed. Parameters: 'key'"
                "\n- 'wait': Wait if needed. Parameters: 'seconds'"
                "\n- 'comment': Include explanations. Parameters: 'comment_text'"
                "\n\nPREFERRED APPROACH: Use the 'find_element' action first, followed by mouse actions."
                "\nThis leverages computer vision to precisely locate UI elements without requiring exact coordinates."
                "\n\nExample plan for 'Click the Start button':"
                "[\n"
                "  {\"action\": \"comment\", \"comment_text\": \"Finding the Start button\"},\n"
                "  {\"action\": \"find_element\", \"description\": \"Start\", \"type\": \"button\"},\n"
                "  {\"action\": \"move_mouse\", \"x\": 0, \"y\": 0},\n"
                "  {\"action\": \"wait\", \"seconds\": 0.5},\n"
                "  {\"action\": \"click\", \"x\": 0, \"y\": 0}\n"
                "]\n"
                "Note: The x,y values after find_element will be automatically updated to the correct coordinates.\n"
                "Ensure the output is ONLY the JSON list, without any surrounding text or markdown."
            )
        else:
            # Regular prompt for initial actions
            system_prompt_content = (
                "You are a helpful assistant translating user commands into computer actions. "
                "Analyze the user's request and respond ONLY with a valid JSON list of action objects. "
                "Rely EXCLUSIVELY on vision-guided actions for navigation within apps. "
                f"\n\n{resolution_info}{standard_resolution}"
                "\n\nSTART EVERY PLAN WITH THESE STEPS:"
                "\n1. Take a screenshot"
                "\n2. Press Windows key to open Start Menu" 
                "\n3. Type the exact app name (like 'discord', 'chrome', etc.)"
                "\n4. Press Enter to launch the app"
                "\n5. Wait for the app to open"
                "\n6. Take another screenshot"
                "\n7. Then continue with mouse movements to navigate"
                "\n\nCore actions:"
                "\n- 'screenshot': Take a screenshot. Parameters: 'filename' (optional)"
                "\n- 'analyze_screen': Analyze the latest screenshot. Parameters: 'description' (optional)"
                "\n- 'type_text': Type text into the active window. Parameters: 'text'"
                "\n- 'press_key': Press a special key. Parameters: 'key' (e.g., enter, tab, esc, ctrl, alt, win, etc.)"
                "\n- 'wait': Pause execution. Parameters: 'seconds'"
                "\n- 'comment': Include a comment in the plan. Parameters: 'comment_text'"
                "\n\nELEMENT DETECTION AND MOUSE ACTIONS (USE AFTER SCREENSHOT ANALYSIS):"
                "\n- 'find_element': Find an element by description. Parameters: 'description', 'type' (optional: 'text', 'button')"
                "\n- 'click': Click at coordinates. Parameters: 'x', 'y'"
                "\n- 'move_mouse': Move the mouse without clicking. Parameters: 'x', 'y', 'duration' (optional)"
                "\n- 'right_click': Right-click at coordinates. Parameters: 'x', 'y'"
                "\n- 'double_click': Double-click at coordinates. Parameters: 'x', 'y'"
                "\n- 'drag': Click and drag from one point to another. Parameters: 'start_x', 'start_y', 'end_x', 'end_y', 'duration' (optional)"
                "\n- 'scroll': Scroll the page. Parameters: 'direction' ('up' or 'down'), 'amount' (optional)"
                "\n\nIMPORTANT RULES:"
                "\n- ALWAYS open applications by pressing Windows key, typing the app name, then pressing Enter"
                "\n- NEVER try to locate app icons in the taskbar or desktop"
                "\n- NEVER include mouse coordinates directly in your initial plan"
                "\n- ALWAYS take a screenshot first, then press Windows key, type app name, press Enter"
                "\n\nExample for 'Open Discord':\n"
                '[\n'
                '  {"action": "comment", "comment_text": "Opening Discord via Windows search"},\n'
                '  {"action": "screenshot"},\n'
                '  {"action": "press_key", "key": "win"},\n'
                '  {"action": "wait", "seconds": 1},\n'
                '  {"action": "type_text", "text": "discord"},\n'
                '  {"action": "wait", "seconds": 1},\n'
                '  {"action": "press_key", "key": "enter"},\n'
                '  {"action": "wait", "seconds": 3},\n'
                '  {"action": "screenshot"}\n'
                ']\n'
                "Note: When using find_element, the coordinates for the following mouse action will be automatically updated.\n"
                "Ensure the output is ONLY the JSON list, without any surrounding text or markdown."
            )
    else: # Use Simple List Format
        if is_vision_follow_up:
            # Special prompt for vision follow-up actions
            system_prompt_content = (
                "You are a computer vision-guided execution assistant. "
                "Based on the screenshot analysis, create a plan of what to do next. "
                "Respond ONLY with a sequence of commands, one per line, prefixed with 'EXECUTE: '. "
                "IMPORTANT: Focus ONLY on mouse movements and interactions based on what you see in the screenshot. "
                "DO NOT suggest taking another screenshot until you've interacted with the elements from the current one. "
                "Be PRECISE with coordinates - use exactly what was found in the analysis. "
                f"\n\n{resolution_info}{standard_resolution}"
                "\n\nFor this follow-up plan, focus on these commands:"
                "\n- find_element(description, type) # Find a UI element; type is optional ('text' or 'button')"
                "\n- move_mouse(x, y) # Move the mouse to an element"
                "\n- click(x, y) # Click on an element"
                "\n- right_click(x, y) # Right-click at coordinates"
                "\n- double_click(x, y) # Double-click at coordinates"
                "\n- drag(start_x, start_y, end_x, end_y) # Drag from one position to another"
                "\n- type_text(text) # Type text if needed"
                "\n- press_key(key) # Press keys if needed"
                "\n- wait(seconds) # Wait if needed"
                "\n- comment(text) # Include explanations"
                "\n\nPREFERRED APPROACH: Use the find_element command first, followed by mouse actions."
                "\nThis leverages computer vision to precisely locate UI elements without requiring exact coordinates."
                "\n\nExample plan for 'Click the Start button':"
                "\nEXECUTE: comment(Finding the Start button)"
                "\nEXECUTE: find_element(Start, button)"
                "\nEXECUTE: move_mouse(0, 0)"
                "\nEXECUTE: wait(0.5)"
                "\nEXECUTE: click(0, 0)"
                "\n\nNote: The coordinates after find_element will be automatically updated."
                "\nEnsure the output ONLY contains EXECUTE: lines."
            )
        else:
            # Regular prompt for initial actions
            system_prompt_content = (
                "You are a helpful assistant translating user commands into computer actions. "
                "Analyze the user's request and respond ONLY with a sequence of specific, "
                "executable commands, one per line, prefixed with 'EXECUTE: '. "
                "Rely EXCLUSIVELY on vision-guided actions - NEVER use direct app launching. "
                f"\n\n{resolution_info}{standard_resolution}"
                "\n\nSTART EVERY PLAN WITH THESE STEPS:"
                "\n1. Take a screenshot"
                "\n2. Analyze the screenshot to identify UI elements"
                "\n3. Then use element detection and mouse movements to navigate"
                "\n\nCore commands:"
                "\n- screenshot() # Takes a screenshot - ALWAYS DO THIS FIRST"
                "\n- analyze_screen(description) # Analyzes the latest screenshot"
                "\n- type_text(text_to_type) # Type text into active window"
                "\n- press_key(key_name) # e.g., enter, tab, win, cmd, ctrl, alt, shift, f1, etc."
                "\n- wait(seconds) # Pause execution"
                "\n- comment(text) # Include explanations for steps"
                "\n\nELEMENT DETECTION AND MOUSE COMMANDS (USE AFTER SCREENSHOT ANALYSIS):"
                "\n- find_element(description, type) # Find a UI element; type is optional ('text' or 'button')"
                "\n- click(x, y) # Click at specific coordinates"
                "\n- move_mouse(x, y) # Moves the mouse without clicking"
                "\n- right_click(x, y) # Right-clicks at coordinates"
                "\n- double_click(x, y) # Double-clicks at coordinates"
                "\n- drag(start_x, start_y, end_x, end_y) # Drag from one position to another"
                "\n- scroll(direction, amount) # Scroll up or down, amount is optional"
                "\n\nIMPORTANT RULES:"
                "\n- To open applications, use Windows key, Start menu, desktop icons, or taskbar via screenshots"
                "\n- NEVER include mouse coordinates directly in your initial plan"
                "\n- ALWAYS take a screenshot first, then analyze it, then interact based on analysis"
                "\n- PREFERRED APPROACH: After analysis, use find_element followed by mouse actions"
                "\n- For the first screenshot, press Windows key to show Start menu"
                "\n\nExample for 'Open Notepad':"
                "\nEXECUTE: comment(Taking a screenshot of the desktop to find a way to open Notepad)"
                "\nEXECUTE: press_key(win)"
                "\nEXECUTE: wait(1)"
                "\nEXECUTE: screenshot()"
                "\nEXECUTE: analyze_screen(Find the search bar or Notepad icon in the Start menu)"
                "\nEXECUTE: find_element(Search, text)"
                "\nEXECUTE: click(0, 0)"
                "\nEXECUTE: type_text(notepad)"
                "\n\nNote: When using find_element, the coordinates for the following mouse action will be automatically updated."
                "\nEnsure the output ONLY contains EXECUTE: lines or comment() lines."
            )
    # --- ---

    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": system_prompt_content
        }
    ]
    
    if is_vision_follow_up:
        # For vision follow-up, include the screenshot analysis as context
        if screenshot_data and screenshot_data.get("analysis"):
            messages.append({
                "role": "user",
                "content": f"Screenshot analysis:\n{screenshot_data['analysis']}\n\nBased on this screenshot analysis, what actions should I take next to accomplish the task: {user_command}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"I need to {user_command}, but the screenshot analysis is not available. Please provide a plan to accomplish this task, starting with taking a screenshot."
            })
    else:
        # Regular user command
        messages.append({
            "role": "user", 
            "content": user_command
        })

        # If there's screenshot data, include it as context but not for follow-up actions
        if screenshot_data and screenshot_data.get("analysis"):
            messages.append({
                "role": "assistant",
                "content": f"I took a screenshot and analyzed it. Here's what I found:\n{screenshot_data['analysis']}"
            })
            messages.append({
                "role": "user",
                "content": "Using the screenshot analysis above, please provide a complete plan to help me with my request."
            })

    # Complete API payload
    payload = {
        "model": OPENMIND_MODEL_ID,
        "messages": messages,
        # Optional parameters (check Openmind docs if needed):
        # "max_tokens": 1000,
        # "temperature": 0.7,
    }

    try:
        print(f"Calling Openmind API: {api_url} with model {OPENMIND_MODEL_ID}...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=240) # Increased timeout from 60 to 120 seconds
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        result = response.json()

        # --- Response Parsing (Based on provided docs) ---
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            interpretation = message.get("content")
            if interpretation:
                 return interpretation.strip()
            else:
                 # Handle potential refusals or empty content
                 refusal = message.get("refusal")
                 if refusal:
                     print(f"Warning: LLM refused to generate content. Refusal data: {refusal}")
                     return f"Error: LLM refused request. Details: {refusal}"
                 else:
                     print(f"Warning: LLM response message content is empty. Full choice: {result['choices'][0]}")
                     return "Error: LLM returned empty content."

        else:
            # Log the unexpected response for debugging
            print(f"Warning: Unexpected API response format. Keys: {result.keys()}")
            print(f"Full Response: {result}")
            return "Error: Could not parse LLM response (unexpected format)."
        # --- ---

    except requests.exceptions.Timeout:
        print("Error: API request timed out.")
        return "Error: API request timed out."
    except requests.exceptions.RequestException as e:
        print(f"Error calling Openmind API: {e}")
        # Try to print more details from the response if available
        error_details = "No additional details."
        if hasattr(e, 'response') and e.response is not None:
            try:
                status = e.response.status_code
                body = e.response.text # Or e.response.json() if you expect JSON errors
                error_details = f"Status Code: {status}, Response Body: {body}"
            except Exception as parse_err:
                 error_details = f"Status Code: {e.response.status_code}. Could not parse response body: {parse_err}"
        print(error_details)
        return f"Error: API request failed. {error_details}"
    except json.JSONDecodeError:
        # This might happen if the response isn't valid JSON (e.g., HTML error page)
        print(f"Error: Could not decode JSON response from API.")
        if 'response' in locals(): # Check if response object exists
             print(f"Raw Response Text: {response.text}")
        return "Error: Invalid JSON response from API."
    except Exception as e:
        # Catch any other unexpected errors during API call/parsing
        import traceback
        print(f"An unexpected error occurred during API interaction: {e}")
        traceback.print_exc() # Print stack trace for debugging
        return "Error: An unexpected error occurred during API interaction."

# --- Screenshot-Based Action Generation Function ---
def get_vision_guided_actions(screenshot_data, original_user_command):
    """
    Generate actions based on a screenshot analysis.
    
    Args:
        screenshot_data: Dictionary with screenshot info
        original_user_command: The user's original command
        
    Returns:
        A plan of actions to execute based on the screenshot
    """
    if not USE_VISION_ANALYSIS:
        print("Vision analysis is disabled. Can't generate vision-guided actions.")
        return None
        
    if not screenshot_data or not screenshot_data.get("path"):
        print("No screenshot data available to generate vision-guided actions.")
        return None
        
    # If we don't have analysis yet, generate it
    if not screenshot_data.get("analysis"):
        analysis = analyze_screenshot("Find interactive elements and their positions")
        screenshot_data["analysis"] = analysis
        print(f"Screenshot analyzed for vision guidance:\n{analysis}")
    
    # Get vision-guided actions from the LLM
    print("Getting vision-guided actions based on screenshot analysis...")
    return get_llm_interpretation(original_user_command, screenshot_data, is_vision_follow_up=True)

# --- Main Function (Updated for Vision-Guided Workflow) ---
def main():
    """
    Main loop for the CUA with action execution and vision guidance.
    """
    print("--- Computer Use Agent with Vision Guidance ---")
    print(f"Using Openmind Provider: {OPENMIND_PROVIDER}, Model: {OPENMIND_MODEL_ID}")
    print(f"Expecting LLM plan format: {'JSON' if USE_JSON_FORMAT else 'Simple List'}")
    print(f"Vision analysis: {'Enabled' if USE_VISION_ANALYSIS else 'Disabled'}")
    print("Enter your command (e.g., 'Open notepad and take a screenshot', or 'quit').")
    print("\n!!! WARNING: This script will attempt to control your mouse and keyboard. !!!")
    print("!!! Supervise closely. Move mouse to a screen corner to stop pyautogui (if failsafe enabled). !!!")
    print("-" * 60)

    # Keep track of current user command for contextual follow-ups
    current_user_command = None

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit", "stop", "q"]:
                print("Exiting CUA.")
                break
            if not user_input.strip():
                continue
                
            # Store the user command for screenshot-based follow-ups
            current_user_command = user_input

            print("\n>>> Sending to Openmind LLM for interpretation and plan...")
            llm_output = get_llm_interpretation(user_input)

            print("-" * 30)
            print(f"LLM Raw Output:\n{llm_output}")
            print("-" * 30)

            # Basic check if the LLM returned an error message
            if llm_output.startswith("Error:"):
                print(f"Cannot execute plan due to LLM error: {llm_output}")
                continue

            # Confirmation before execution (RECOMMENDED FOR SAFETY)
            confirm = input("Proceed with execution? (y/n): ")
            if confirm.lower() != 'y':
                print("Execution cancelled by user.")
                print("-" * 60)
                continue

            # Give user time to switch focus if needed, or prepare
            print("Executing in 3 seconds... (Switch focus NOW if needed)")
            time.sleep(1)
            print("Executing in 2 seconds...")
            time.sleep(1)
            print("Executing in 1 second...")
            time.sleep(1)

            # Define a callback function for vision-guided actions after screenshots
            def vision_callback(screenshot_data):
                print("Screenshot taken, generating vision-guided actions...")
                return get_vision_guided_actions(screenshot_data, current_user_command)

            # Execute the plan based on the chosen format, with vision guidance
            if USE_JSON_FORMAT:
                execute_plan_json(llm_output, vision_callback if USE_VISION_ANALYSIS else None)
            else:
                execute_plan_simple_list(llm_output, vision_callback if USE_VISION_ANALYSIS else None)
                
            print("-" * 60)
            print(">>> Task attempt finished.")
            print("-" * 60)

        except KeyboardInterrupt:
             print("\nExecution interrupted by user (Ctrl+C). Exiting.")
             break
        except Exception as e:
            # Catch errors in the main loop itself
            import traceback
            print(f"\n--- A critical error occurred in the main loop: {e} ---")
            traceback.print_exc()
            print("Continuing...")
            time.sleep(1)


if __name__ == "__main__":
    # Check if pyautogui is available early
    try:
        import pyautogui
    except ImportError:
        print("Error: PyAutoGUI library not found.")
        print("Please install it using: pip install pyautogui")
        exit()
    
    # Check for other required packages
    try:
        import PIL
        import requests
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("Please install missing packages using pip:")
        print("pip install pillow requests python-dotenv")
        exit()

    main()