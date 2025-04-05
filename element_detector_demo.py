#!/usr/bin/env python
# element_detector_demo.py - Demonstrates the enhanced element detection capabilities

import os
import sys
import time
import argparse
from element_detector import ElementDetector, find_ui_elements, get_element_coordinates, describe_element_location, TESSERACT_AVAILABLE

def take_screenshot():
    """Take a screenshot and save it to a temporary file."""
    import pyautogui
    import tempfile
    
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(tempfile.gettempdir(), "cua_element_detection")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a filename with timestamp
    timestamp = int(time.time())
    filename = os.path.join(temp_dir, f"screenshot_{timestamp}.png")
    
    # Take the screenshot
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)
    
    print(f"Screenshot saved to: {filename}")
    return filename

def demo_element_detection(screenshot_path=None, find_text=None, find_type=None, highlight=True):
    """
    Demonstrate the element detection capabilities.
    
    Args:
        screenshot_path: Path to existing screenshot or None to take a new one
        find_text: Optional text to search for
        find_type: Optional element type to find (text, button)
        highlight: Whether to create a highlighted debug image
    """
    # Take a screenshot if not provided
    if not screenshot_path:
        screenshot_path = take_screenshot()
        
    # Display OCR availability warning if needed
    if not TESSERACT_AVAILABLE and find_text:
        print("WARNING: Tesseract OCR is not installed. Text detection will be limited.")
        print("Consider installing Tesseract OCR for better text element detection.")
        print("See README.md for installation instructions.")
        print("Proceeding with button detection only...")
        
    # Give the user a moment to see what was captured
    print("Analyzing screenshot...")
    time.sleep(1)
    
    # Create element detector
    detector = ElementDetector()
    detector.load_screenshot(screenshot_path)
    
    # Find all UI elements
    if find_text or find_type:
        # Specific search
        print(f"Searching for elements matching: Text='{find_text}', Type='{find_type}'")
        elements = find_ui_elements(screenshot_path, find_type, find_text)
        
        if not elements:
            print(f"No elements found matching the criteria.")
            return
            
        print(f"Found {len(elements)} matching elements:")
        for i, elem in enumerate(elements):
            print(f"\nElement {i+1}:")
            print(f"  Type: {elem.get('type', 'unknown')}")
            if 'text' in elem:
                print(f"  Text: '{elem['text']}'")
            
            coords = get_element_coordinates(elem)
            if coords:
                print(f"  Coordinates: {coords}")
                
            print(f"  Description: {describe_element_location(elem)}")
            
        # Highlight matching elements
        if highlight and elements:
            debug_path = detector.highlight_elements(elements)
            print(f"Debug image with highlighted elements saved to: {debug_path}")
            
    else:
        # Comprehensive detection
        print("Performing comprehensive element detection...")
        results = detector.detect_all(screenshot_path)
        
        if results["success"]:
            print("\nDetection Results:")
            
            # If OCR is not available, show a message about text elements
            if not TESSERACT_AVAILABLE:
                print("- Text detection is disabled (Tesseract OCR not installed)")
            else:
                print(f"- Found {results['count']['text']} text elements")
                
            print(f"- Found {results['count']['buttons']} button-like elements")
            print(f"- Total: {results['count']['total']} UI elements")
            
            # Show sample of text elements
            if results["elements"]["text"]:
                print("\nSample Text Elements:")
                for i, elem in enumerate(results["elements"]["text"][:5]):  # Show first 5
                    if 'text' in elem and elem['text'].strip():
                        coords = get_element_coordinates(elem)
                        if coords:
                            print(f"  - '{elem['text']}' at {coords}")
                
                if len(results["elements"]["text"]) > 5:
                    print(f"  ... and {len(results['elements']['text']) - 5} more text elements")
            
            # Show debug image path
            if results.get("debug_image"):
                print(f"\nDebug image saved to: {results['debug_image']}")
                
    print("\nElement detection demo completed.")

def interactive_demo():
    """Run an interactive element finding demo."""
    print("=== Interactive Element Detector Demo ===")
    print("This demo will let you take a screenshot and find UI elements.")
    
    # Display OCR availability information
    if not TESSERACT_AVAILABLE:
        print("\nWARNING: Tesseract OCR is not installed. Text detection will be limited.")
        print("Consider installing Tesseract OCR for better text element detection.")
        print("See README.md for installation instructions.")
        print("Proceeding with button detection only...\n")
    
    # Take initial screenshot
    screenshot_path = take_screenshot()
    
    while True:
        print("\nOptions:")
        print("1. Take a new screenshot")
        print("2. Find elements by text" + (" (Requires Tesseract OCR)" if not TESSERACT_AVAILABLE else ""))
        print("3. Find button-like elements")
        print("4. Detect all elements")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            screenshot_path = take_screenshot()
        elif choice == '2':
            if not TESSERACT_AVAILABLE:
                print("Warning: Text detection requires Tesseract OCR which is not installed.")
                print("Results may be limited. See README.md for installation instructions.")
            text = input("Enter text to search for: ").strip()
            demo_element_detection(screenshot_path, find_text=text, find_type='text')
        elif choice == '3':
            demo_element_detection(screenshot_path, find_type='button')
        elif choice == '4':
            demo_element_detection(screenshot_path)
        elif choice == '5':
            print("Exiting demo.")
            break
        else:
            print("Invalid choice. Please enter a number from 1-5.")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Element Detector Demo")
    parser.add_argument("--screenshot", help="Path to existing screenshot to analyze")
    parser.add_argument("--find-text", help="Find elements containing this text")
    parser.add_argument("--find-type", choices=["text", "button"], help="Find elements of this type")
    parser.add_argument("--interactive", action="store_true", help="Run interactive demo")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Set debug mode if requested
    if args.debug:
        os.environ['CUA_DEBUG'] = 'true'
    
    # Display OCR status
    if not TESSERACT_AVAILABLE and (args.find_text or not args.find_type):
        print("WARNING: Tesseract OCR is not installed. Text detection will be limited.")
        print("Consider installing Tesseract OCR for better text element detection.")
        print("See README.md for installation instructions.")
        
    # Run the demo
    if args.interactive:
        interactive_demo()
    else:
        demo_element_detection(args.screenshot, args.find_text, args.find_type) 