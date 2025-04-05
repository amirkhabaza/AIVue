# Computer Use Agent (CUA) with Vision Guidance

A Python-based agent that can control your computer through vision-guided interaction, responding to natural language commands by translating them into precise mouse and keyboard actions.

## Features

### Fully Vision-Guided

The CUA uses a screenshot-first approach to interact with your computer, analyzing the screen to identify UI elements before taking any actions. This allows for natural computer interaction:

- **Vision Analysis**: Takes screenshots and analyzes them to understand the screen layout
- **Element Detection**: Uses advanced computer vision to precisely identify UI elements
- **UI Navigation**: Interacts with elements by finding and clicking instead of hardcoded coordinates
- **Contextual Actions**: Makes decisions based on real-time screen content

### Element Detection Technology

The system now includes an advanced element detector that provides more precise identification of UI elements:

- **Computer Vision-Based**: Uses OpenCV and image processing to identify buttons, text fields and other UI elements
- **OCR Integration**: Leverages Tesseract OCR to read and locate text on screen
- **Precise Coordinates**: Pinpoints exact element locations for reliable interaction
- **Element Memory**: Tracks elements across interactions for consistent reference
- **Debug Visualization**: Can generate highlighted screenshots showing detected elements

### Screenshot-First Approach

Every interaction follows a structured workflow:

1. **Take a Screenshot**: Captures the current screen state
2. **Analyze with Vision**: Identifies UI elements, text, and interactive regions
3. **Element Detection**: Precisely locates buttons, text fields, and other UI elements
4. **Navigation Actions**: Uses mouse and keyboard to interact with identified elements
5. **Feedback Loop**: Takes additional screenshots to verify results and determine next steps

## Requirements

- Python 3.8 or higher
- OpenAI API key (for GPT-4 Vision API access)
- Required libraries:
  - pyautogui (for mouse/keyboard control)
  - OpenCV (for computer vision element detection) 
  - pytesseract (for OCR capabilities)
  - Pillow (for image processing)
  - requests (for API communication)
  - python-dotenv (for environment variable management)

## Setup

1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Obtain an API key from OpenAI (or another provider with vision capabilities)
4. Set up environment variables:
   - Create a `.env` file in the project directory
   - Add `OPENMIND_API_KEY=your_api_key_here` to the file
   - Optionally, if using Windows, set `TESSERACT_PATH` to point to your Tesseract OCR installation

### Installing Tesseract OCR

For full text detection capabilities, you need to install Tesseract OCR:

#### Windows:
1. Download the installer from [UB Mannheim's GitHub page](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer and remember the installation path (default is `C:\Program Files\Tesseract-OCR`)
3. Set the TESSERACT_PATH environment variable either:
   - In your `.env` file: `TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe`
   - Or as a system environment variable

#### macOS:
```bash
brew install tesseract
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
```

#### Verifying Installation:
To verify Tesseract is installed correctly, run:
```bash
tesseract --version
```

If you don't install Tesseract OCR, the system will still work using basic button detection, but text-based element finding will be disabled.

## Usage

Run the main script:

```bash
python AGI.py
```

Then enter natural language commands like:
- "Open the Chrome browser"
- "Find and click on the settings icon"
- "Take a screenshot and find the login button"

### Example Commands

- **Basic operations:** "Open the Notepad app and type 'Hello World'"
- **Complex workflows:** "Download the latest version of Python from python.org"
- **UI Navigation:** "Find the login button and enter my credentials"
- **Element Finding:** "Find the search box, type 'weather', and press Enter"

## How Element Detection Works

The new element detection system works in multiple stages:

1. **Screenshot Analysis**: The initial screenshot is processed using OpenCV techniques
2. **Text Detection**: OCR identifies and locates all text elements with their coordinates
3. **Button Detection**: Computer vision algorithms identify button-like shapes and elements
4. **Element Mapping**: All detected elements are mapped with precise coordinates
5. **Intelligent Selection**: When given a description like "Find the login button", the system matches it to the best detected element
6. **Coordinate Injection**: Coordinates are automatically injected into subsequent mouse actions

### Using the Find Element Action

The system now supports a new action:

```json
{"action": "find_element", "description": "Login", "type": "button"}
```

This action will:
1. Search for UI elements matching the description
2. Locate the element's exact coordinates
3. Automatically update the coordinates for the next mouse action

This provides a more reliable way to interact with UI elements without relying on potentially inaccurate vision model coordinate estimations.

## Safety Notes

- **⚠️ WARNING:** This script controls your mouse and keyboard.
- Always be ready to interrupt execution by moving your mouse to a screen corner (failsafe)
- Review plans before confirming execution
- Use in a controlled environment initially

## Technical Details

The system consists of three main components:
- `AGI.py`: Main script that processes user commands and calls the API
- `action_executor.py`: Executes the generated action plan through mouse/keyboard control
- `element_detector.py`: Provides advanced computer vision for precise UI element detection

The element detector uses:
- OpenCV for image processing and element shape detection
- Tesseract OCR for text recognition
- Custom algorithms for correlating visual elements with functionality

## Troubleshooting

If you encounter issues with element detection:
- Ensure Tesseract OCR is properly installed and configured
- Try running with debug mode enabled: `CUA_DEBUG=true python AGI.py`
- Check the generated debug images in the temp directory to see what elements are being detected

### Common Errors:

1. **"Tesseract OCR is not properly installed or configured"**:
   - Make sure you've installed Tesseract OCR following the instructions above
   - Verify the installation path is correct
   - Try setting the TESSERACT_PATH environment variable explicitly
   - If you can't install Tesseract, the system will still function but with limited text recognition

2. **"Error clicking at (0, 0): PyAutoGUI fail-safe triggered"**:
   - This happens when coordinates are (0,0) and PyAutoGUI's failsafe is triggered
   - Make sure the element detection is working by checking debug images
   - Try disabling PyAutoGUI's failsafe in action_executor.py (not recommended for normal use)

3. **No elements found matching 'X'**:
   - Try using a more general element type (e.g., use "type": "button" instead of "type": "text")
   - Check if the element is visible in the screenshot
   - Try running with debug mode to see what elements are being detected 