# element_detector.py
import cv2
import numpy as np
import pytesseract
import pyautogui
import os
import sys
import time
from PIL import Image
import tempfile
import json

# Configure pytesseract path for Windows (if needed)
TESSERACT_AVAILABLE = True
try:
    if sys.platform == "win32":
        # Update this path to your Tesseract installation if needed
        if os.getenv('TESSERACT_PATH'):
            pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH')
        else:
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # Test if Tesseract is available by running a simple OCR test
    pytesseract.get_tesseract_version()
except Exception as e:
    print(f"Warning: Tesseract OCR is not properly installed or configured: {e}")
    print("OCR-based text detection will be disabled. Only basic button detection will be available.")
    print("To enable full functionality, please install Tesseract OCR and set TESSERACT_PATH.")
    TESSERACT_AVAILABLE = False

# Constants
TEMP_DIR = os.path.join(tempfile.gettempdir(), "cua_element_detection")
os.makedirs(TEMP_DIR, exist_ok=True)

# Default target resolution
DEFAULT_SCREEN_WIDTH = 1920
DEFAULT_SCREEN_HEIGHT = 1080

# Debug mode
DEBUG = os.getenv('CUA_DEBUG', 'False').lower() == 'true'

class ElementDetector:
    """Class for detecting UI elements in screenshots using computer vision techniques."""
    
    def __init__(self):
        """Initialize the element detector with default parameters."""
        self.last_screenshot_path = None
        self.last_screenshot_image = None
        self.last_screenshot_gray = None
        self.last_elements = []
        self.screen_width, self.screen_height = pyautogui.size()
        # Calculate scale factors relative to default resolution
        self.width_scale = self.screen_width / DEFAULT_SCREEN_WIDTH
        self.height_scale = self.screen_height / DEFAULT_SCREEN_HEIGHT
        
        if DEBUG and (self.width_scale != 1 or self.height_scale != 1):
            print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
            print(f"Scale factors: width={self.width_scale:.2f}, height={self.height_scale:.2f}")
    
    def normalize_coordinates(self, x, y, reverse=False):
        """
        Normalize coordinates between default resolution and actual screen resolution.
        
        Args:
            x: X coordinate
            y: Y coordinate
            reverse: If True, convert from actual screen to default resolution coordinates
                    If False (default), convert from default to actual screen coordinates
        
        Returns:
            Tuple (x, y) with normalized coordinates
        """
        if reverse:
            # Convert from actual screen coordinates to default resolution
            normalized_x = int(x / self.width_scale)
            normalized_y = int(y / self.height_scale)
        else:
            # Convert from default resolution to actual screen coordinates
            normalized_x = int(x * self.width_scale)
            normalized_y = int(y * self.height_scale)
            
        return normalized_x, normalized_y
    
    def load_screenshot(self, screenshot_path):
        """Load a screenshot for analysis.
        
        Args:
            screenshot_path: Path to the screenshot image file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Read the image
            self.last_screenshot_path = screenshot_path
            self.last_screenshot_image = cv2.imread(screenshot_path)
            
            if self.last_screenshot_image is None:
                print(f"Error: Could not read image at {screenshot_path}")
                return False
                
            # Convert to grayscale for processing
            self.last_screenshot_gray = cv2.cvtColor(self.last_screenshot_image, cv2.COLOR_BGR2GRAY)
            return True
        except Exception as e:
            print(f"Error loading screenshot: {e}")
            return False
    
    def find_text_elements(self, min_confidence=80):
        """Find text elements in the screenshot using OCR.
        
        Args:
            min_confidence: Minimum confidence score (0-100) to consider a text detection valid
            
        Returns:
            List of dictionaries containing text elements with coordinates
        """
        if self.last_screenshot_path is None:
            print("Error: No screenshot loaded")
            return []
            
        if not TESSERACT_AVAILABLE:
            print("Skipping text detection as Tesseract OCR is not available")
            return []
            
        try:
            # Use pytesseract to get text and coordinates
            ocr_data = pytesseract.image_to_data(
                Image.open(self.last_screenshot_path), 
                output_type=pytesseract.Output.DICT,
                config='--oem 3 --psm 11'
            )
            
            text_elements = []
            for i in range(len(ocr_data['text'])):
                # Filter out empty text and low confidence detections
                if not ocr_data['text'][i].strip() or int(ocr_data['conf'][i]) < min_confidence:
                    continue
                    
                # Get bounding box
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Calculate center point
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Normalize the coordinates for consistent targeting
                if self.width_scale != 1 or self.height_scale != 1:
                    # Convert coordinates from screenshot to standard scale
                    # Note: We use reverse=True because screenshot coordinates need to be 
                    # converted to the standardized coordinate space
                    std_x, std_y = self.normalize_coordinates(center_x, center_y, reverse=True)
                    
                    # Then convert to actual screen coordinates
                    screen_x, screen_y = self.normalize_coordinates(std_x, std_y, reverse=False)
                    
                    if DEBUG:
                        print(f"Text element '{ocr_data['text'][i]}': Original coords ({center_x}, {center_y}), " 
                              f"Normalized to ({screen_x}, {screen_y})")
                else:
                    screen_x, screen_y = center_x, center_y
                
                text_elements.append({
                    'type': 'text',
                    'text': ocr_data['text'][i],
                    'confidence': int(ocr_data['conf'][i]),
                    'x': screen_x,
                    'y': screen_y,
                    'width': w,
                    'height': h,
                    'bbox': (x, y, x+w, y+h)
                })
            
            return text_elements
        except Exception as e:
            print(f"Error finding text elements: {e}")
            return []
    
    def find_ui_buttons(self):
        """Find button-like UI elements in the screenshot.
        
        Returns:
            List of dictionaries containing button elements with coordinates
        """
        if self.last_screenshot_image is None:
            print("Error: No screenshot loaded")
            return []
            
        try:
            # Make a copy of the image to draw on for debugging
            debug_image = self.last_screenshot_image.copy() if DEBUG else None
            
            # Convert to grayscale if not already
            if self.last_screenshot_gray is None:
                gray = cv2.cvtColor(self.last_screenshot_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.last_screenshot_gray
                
            # Apply binary thresholding to isolate potential UI elements
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Apply morphological operations to enhance button shapes
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=1)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            ui_elements = []
            for i, contour in enumerate(contours):
                # Filter out very small contours
                if cv2.contourArea(contour) < 500:  # Minimum area threshold
                    continue
                    
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter out elements that are too large (likely background)
                if w > self.screen_width * 0.5 or h > self.screen_height * 0.5:
                    continue
                    
                # Aspect ratio check for button-like elements
                aspect_ratio = float(w) / h
                if aspect_ratio > 5 or aspect_ratio < 0.2:  # Skip very narrow or wide elements
                    continue
                
                # Calculate center point
                center_x = x + w // 2
                center_y = y + h // 2
                
                ui_elements.append({
                    'type': 'button',
                    'x': center_x,
                    'y': center_y,
                    'width': w,
                    'height': h,
                    'bbox': (x, y, x+w, y+h),
                    'area': cv2.contourArea(contour)
                })
                
                # Draw on debug image
                if DEBUG:
                    cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(debug_image, f"Button {i}", (x, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save debug image
            if DEBUG:
                debug_path = os.path.join(TEMP_DIR, "button_detection_debug.png")
                cv2.imwrite(debug_path, debug_image)
                print(f"Debug image saved to {debug_path}")
            
            return ui_elements
        except Exception as e:
            print(f"Error finding UI buttons: {e}")
            return []
    
    def find_clickable_regions(self):
        """Identify potentially clickable regions by combining text and button detection.
        
        Returns:
            List of dictionaries containing all detected elements with coordinates
        """
        all_elements = []
        
        # Find text elements
        text_elements = self.find_text_elements()
        all_elements.extend(text_elements)
        
        # Find button-like elements
        button_elements = self.find_ui_buttons()
        all_elements.extend(button_elements)
        
        # Store for later reference
        self.last_elements = all_elements
        
        return all_elements
    
    def find_element_by_text(self, text, fuzzy_match=True, case_sensitive=False):
        """Find an element containing the specified text.
        
        Args:
            text: Text to search for
            fuzzy_match: Whether to use partial matching
            case_sensitive: Whether to consider case in matching
            
        Returns:
            Dictionary with element info or None if not found
        """
        if not self.last_elements:
            # Try to find elements if none are cached
            self.find_clickable_regions()
            
        if not case_sensitive:
            search_text = text.lower()
        else:
            search_text = text
            
        best_match = None
        for element in self.last_elements:
            if 'text' not in element:
                continue
                
            element_text = element['text']
            if not case_sensitive:
                element_text = element_text.lower()
                
            if fuzzy_match:
                if search_text in element_text:
                    if best_match is None or len(element_text) < len(best_match['text']):
                        best_match = element
            else:
                if search_text == element_text:
                    return element
        
        return best_match
    
    def find_element_by_type(self, element_type):
        """Find elements of a specific type.
        
        Args:
            element_type: Type of element to find ('text', 'button', etc.)
            
        Returns:
            List of matching elements
        """
        if not self.last_elements:
            # Try to find elements if none are cached
            self.find_clickable_regions()
            
        return [elem for elem in self.last_elements if elem.get('type') == element_type]
    
    def find_element_at_position(self, x, y, tolerance=10):
        """Find the element at or near the specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            tolerance: Pixel tolerance for matching
            
        Returns:
            Element dictionary or None if not found
        """
        if not self.last_elements:
            # Try to find elements if none are cached
            self.find_clickable_regions()
            
        for element in self.last_elements:
            # Check if point is within the bounding box + tolerance
            if 'bbox' in element:
                x1, y1, x2, y2 = element['bbox']
                if (x1 - tolerance <= x <= x2 + tolerance and 
                    y1 - tolerance <= y <= y2 + tolerance):
                    return element
                    
            # Alternatively check center point
            elif 'x' in element and 'y' in element:
                elem_x, elem_y = element['x'], element['y']
                distance = ((x - elem_x) ** 2 + (y - elem_y) ** 2) ** 0.5
                if distance <= tolerance:
                    return element
                    
        return None

    def highlight_elements(self, elements=None, output_path=None):
        """Create a debug image highlighting detected elements.
        
        Args:
            elements: List of elements to highlight (uses last_elements if None)
            output_path: Path to save the output image
            
        Returns:
            Path to the saved debug image
        """
        if self.last_screenshot_image is None:
            print("Error: No screenshot loaded")
            return None
            
        if elements is None:
            elements = self.last_elements
            
        if not elements:
            print("No elements to highlight")
            return None
            
        # Create a copy of the image to draw on
        debug_image = self.last_screenshot_image.copy()
        
        # Colors for different element types
        colors = {
            'text': (0, 255, 0),    # Green
            'button': (0, 0, 255),  # Red
            'default': (255, 0, 0)  # Blue
        }
        
        # Draw each element
        for i, element in enumerate(elements):
            element_type = element.get('type', 'default')
            color = colors.get(element_type, colors['default'])
            
            # Add normalized coordinate information to the debug image
            if 'x' in element and 'y' in element:
                x, y = element['x'], element['y']
                
                # Convert screen coordinates to image coordinates for drawing
                if self.width_scale != 1 or self.height_scale != 1:
                    # We need to reverse the normalization to get image coordinates
                    img_x, img_y = self.normalize_coordinates(x, y, reverse=True)
                else:
                    img_x, img_y = x, y
                
                # Ensure we don't draw outside image bounds
                if 0 <= img_x < debug_image.shape[1] and 0 <= img_y < debug_image.shape[0]:
                    # Draw point
                    cv2.circle(debug_image, (img_x, img_y), 5, color, -1)
                    
                    # Draw coordinates text
                    coord_text = f"({x}, {y})"
                    cv2.putText(debug_image, coord_text, (img_x + 10, img_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw bounding box if available
            if 'bbox' in element:
                x1, y1, x2, y2 = element['bbox']
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
            
            # Add text label
            label = f"{element_type}"
            if 'text' in element:
                label += f": {element['text']}"
                
            if 'x' in element and 'y' in element:
                # Use the image coordinates for drawing
                if self.width_scale != 1 or self.height_scale != 1:
                    img_x, img_y = self.normalize_coordinates(element['x'], element['y'], reverse=True)
                else:
                    img_x, img_y = element['x'], element['y']
                
                if 0 <= img_x < debug_image.shape[1] and 0 <= img_y < debug_image.shape[0]:
                    cv2.putText(debug_image, label, (img_x, img_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save the debug image
        if output_path is None:
            output_path = os.path.join(TEMP_DIR, "highlighted_elements.png")
            
        cv2.imwrite(output_path, debug_image)
        print(f"Debug image with highlighted elements saved to {output_path}")
        return output_path

    def match_template(self, template_path, threshold=0.8):
        """Find instances of a template image in the screenshot.
        
        Args:
            template_path: Path to the template image
            threshold: Matching threshold (0.0-1.0)
            
        Returns:
            List of dictionaries with match locations and scores
        """
        if self.last_screenshot_image is None:
            print("Error: No screenshot loaded")
            return []
            
        try:
            # Read the template
            template = cv2.imread(template_path)
            if template is None:
                print(f"Error: Could not read template image at {template_path}")
                return []
                
            # Convert template to grayscale
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # Perform template matching
            result = cv2.matchTemplate(self.last_screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            
            # Get matches above threshold
            locations = np.where(result >= threshold)
            matches = []
            
            template_h, template_w = template_gray.shape
            
            # Create debug image if needed
            debug_image = self.last_screenshot_image.copy() if DEBUG else None
            
            for pt in zip(*locations[::-1]):
                # Calculate center point
                center_x = pt[0] + template_w // 2
                center_y = pt[1] + template_h // 2
                
                # Get the match score at this location
                score = result[pt[1], pt[0]]
                
                matches.append({
                    'type': 'template_match',
                    'template': os.path.basename(template_path),
                    'score': float(score),
                    'x': center_x,
                    'y': center_y,
                    'width': template_w,
                    'height': template_h,
                    'bbox': (pt[0], pt[1], pt[0] + template_w, pt[1] + template_h)
                })
                
                # Draw on debug image
                if DEBUG:
                    cv2.rectangle(debug_image, pt, (pt[0] + template_w, pt[1] + template_h), (0, 255, 255), 2)
                    cv2.putText(debug_image, f"{score:.2f}", (pt[0], pt[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Save debug image
            if DEBUG and matches:
                debug_path = os.path.join(TEMP_DIR, "template_matching_debug.png")
                cv2.imwrite(debug_path, debug_image)
                print(f"Debug image saved to {debug_path}")
            
            return matches
        except Exception as e:
            print(f"Error matching template: {e}")
            return []

    def save_element_template(self, element, output_path=None):
        """Extract an element from the screenshot and save it as a template.
        
        Args:
            element: Element dictionary with bbox or x,y,width,height
            output_path: Path to save the template image
            
        Returns:
            Path to the saved template
        """
        if self.last_screenshot_image is None:
            print("Error: No screenshot loaded")
            return None
            
        try:
            # Get element bounds
            if 'bbox' in element:
                x1, y1, x2, y2 = element['bbox']
                width, height = x2 - x1, y2 - y1
            elif all(k in element for k in ['x', 'y', 'width', 'height']):
                center_x, center_y = element['x'], element['y']
                width, height = element['width'], element['height']
                x1 = center_x - width // 2
                y1 = center_y - height // 2
            else:
                print("Error: Element has no bbox or x,y,width,height")
                return None
                
            # Ensure coordinates are valid
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.last_screenshot_image.shape[1], x1 + width)
            y2 = min(self.last_screenshot_image.shape[0], y1 + height)
            
            # Extract the element
            template = self.last_screenshot_image[y1:y2, x1:x2]
            
            # Generate output path if not provided
            if output_path is None:
                element_type = element.get('type', 'element')
                element_text = element.get('text', '')
                if element_text:
                    element_text = element_text.replace(' ', '_').lower()
                    filename = f"{element_type}_{element_text}.png"
                else:
                    filename = f"{element_type}_{int(time.time())}.png"
                output_path = os.path.join(TEMP_DIR, filename)
            
            # Save the template
            cv2.imwrite(output_path, template)
            print(f"Element template saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving element template: {e}")
            return None

    def detect_all(self, screenshot_path):
        """Perform comprehensive element detection on a screenshot.
        
        Args:
            screenshot_path: Path to the screenshot image
            
        Returns:
            Dictionary with all detected elements
        """
        if not self.load_screenshot(screenshot_path):
            return {"success": False, "error": "Failed to load screenshot"}
            
        # Detect all types of elements
        text_elements = self.find_text_elements()
        button_elements = self.find_ui_buttons()
        
        # Combine all elements
        all_elements = text_elements + button_elements
        self.last_elements = all_elements
        
        # Create a highlighted debug image
        debug_image_path = None
        if DEBUG:
            debug_image_path = self.highlight_elements(all_elements)
            
        # Return structured data
        return {
            "success": True,
            "screenshot_path": screenshot_path,
            "image_width": self.last_screenshot_image.shape[1],
            "image_height": self.last_screenshot_image.shape[0],
            "elements": {
                "text": text_elements,
                "buttons": button_elements,
                "all": all_elements
            },
            "count": {
                "text": len(text_elements),
                "buttons": len(button_elements),
                "total": len(all_elements)
            },
            "debug_image": debug_image_path
        }

# Helper Functions

def find_ui_elements(screenshot_path, element_type=None, text=None):
    """Convenience function to find UI elements in a screenshot.
    
    Args:
        screenshot_path: Path to the screenshot image
        element_type: Optional type of element to find ('text', 'button', etc.)
        text: Optional text to search for
        
    Returns:
        List of matching elements
    """
    detector = ElementDetector()
    detector.load_screenshot(screenshot_path)
    
    # Detect all elements
    detector.find_clickable_regions()
    
    # Filter by type if specified
    if element_type:
        elements = detector.find_element_by_type(element_type)
    else:
        elements = detector.last_elements
        
    # Filter by text if specified
    if text:
        if element_type:
            # Find matching element of specific type
            matching = [elem for elem in elements 
                        if 'text' in elem and text.lower() in elem['text'].lower()]
            return matching
        else:
            # Find any element with matching text
            element = detector.find_element_by_text(text)
            return [element] if element else []
    
    return elements

def save_detection_results(results, output_path=None):
    """Save detection results to a JSON file.
    
    Args:
        results: Detection results dictionary
        output_path: Path to save the JSON file
        
    Returns:
        Path to the saved JSON file
    """
    if output_path is None:
        timestamp = int(time.time())
        output_path = os.path.join(TEMP_DIR, f"detection_results_{timestamp}.json")
        
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detection results saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving detection results: {e}")
        return None

def get_element_coordinates(element):
    """Extract coordinates from an element dictionary.
    
    Args:
        element: Element dictionary
        
    Returns:
        Tuple of (x, y) coordinates or None if not available
    """
    if not element:
        return None
        
    if 'x' in element and 'y' in element:
        return (element['x'], element['y'])
    elif 'bbox' in element:
        x1, y1, x2, y2 = element['bbox']
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    return None

def describe_element_location(element, image_width=None, image_height=None):
    """Generate a human-readable description of an element's location.
    
    Args:
        element: Element dictionary
        image_width: Optional width of the full image
        image_height: Optional height of the full image
        
    Returns:
        String description of the element's location
    """
    if not element:
        return "Element not found"
        
    # Get coordinates
    coords = get_element_coordinates(element)
    if not coords:
        return "Element position unknown"
        
    x, y = coords
    
    # Basic position description
    description = f"Located at coordinates ({x}, {y})"
    
    # Add relative position if image dimensions are provided
    if image_width and image_height:
        # Horizontal position
        if x < image_width * 0.33:
            h_pos = "left side"
        elif x < image_width * 0.66:
            h_pos = "center"
        else:
            h_pos = "right side"
            
        # Vertical position
        if y < image_height * 0.33:
            v_pos = "top"
        elif y < image_height * 0.66:
            v_pos = "middle"
        else:
            v_pos = "bottom"
            
        description += f" on the {v_pos} {h_pos} of the screen"
        
    # Add element type and text if available
    if 'type' in element:
        description += f", element type: {element['type']}"
        
    if 'text' in element and element['text']:
        description += f", text: '{element['text']}'"
        
    return description

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="UI Element Detector")
    parser.add_argument("--screenshot", required=True, help="Path to screenshot image")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--output", help="Path to save detection results")
    parser.add_argument("--find-text", help="Find elements containing this text")
    parser.add_argument("--find-type", choices=["text", "button"], help="Find elements of this type")
    
    args = parser.parse_args()
    
    # Set debug mode
    if args.debug:
        DEBUG = True
    
    # Create detector
    detector = ElementDetector()
    
    # Process screenshot
    if args.find_text or args.find_type:
        # Specific search
        elements = find_ui_elements(args.screenshot, args.find_type, args.find_text)
        
        # Print results
        print(f"Found {len(elements)} matching elements")
        for i, elem in enumerate(elements):
            coords = get_element_coordinates(elem)
            if coords:
                print(f"Element {i+1}: {describe_element_location(elem)}")
            
        # Highlight matches
        if elements:
            detector.highlight_elements(elements)
    else:
        # Full detection
        results = detector.detect_all(args.screenshot)
        
        # Print summary
        if results["success"]:
            print(f"Detected {results['count']['total']} elements:")
            print(f"- {results['count']['text']} text elements")
            print(f"- {results['count']['buttons']} button elements")
            
            # Save results if requested
            if args.output:
                save_detection_results(results, args.output)
        else:
            print(f"Detection failed: {results.get('error', 'Unknown error')}") 