#!/usr/bin/env python3
"""
Dynamic Completion Detection Module
Extracted from working_ui_agent.py for use in demo.py and deployment system
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from typing import Optional, Dict, Any, Callable
from download_model import download_model

class BinaryUI(nn.Module):
    """
    Binary UI classifier from AdaT - determines if UI rendering is complete
    """
    def __init__(self, model_path=None, model_name='mobilenet'):
        super(BinaryUI, self).__init__()
        
        # Load pre-trained model
        if model_name == 'mobilenet':
            self.model = models.mobilenet_v2(pretrained=False)
            # Modify for binary classification
            self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Load trained weights if provided
        if model_path and os.path.exists(model_path):
            try:
                self.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"✅ Loaded UI completion model from {model_path}")
            except Exception as e:
                print(f"⚠️  Failed to load model: {e}")
        else:
            print("⚠️  No model path provided or model doesn't exist")
        
        self.eval()
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, image):
        """Predict if UI rendering is complete (0=blurred, 1=rendered)"""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif isinstance(image, str):
                # If it's a file path
                image = Image.open(image).convert('RGB')
            
            input_tensor = self.preprocess(image)
            input_batch = input_tensor.unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            return probabilities[1].item()  # Probability that UI is fully rendered
        except Exception as e:
            print(f"⚠️  UI completion prediction failed: {e}")
            return 0.5  # Default value


class AdaptiveWait:
    """
    Adaptive waiting system using AdaT's binary classification approach
    """
    
    def __init__(self, 
                 model_path=None,
                 max_model_tries=4,
                 max_throttle_ms=1000,
                 min_wait_time=0.5,
                 max_wait_time=5.0):
        
        # Try to download model if not provided
        if model_path is None:
            try:
                model_path = download_model()
                print(f"📥 Downloaded model for completion detection: {model_path}")
            except Exception as e:
                print(f"⚠️  Failed to download model: {e}")
        
        self.classifier = BinaryUI(model_path)
        self.max_model_tries = max_model_tries
        self.max_throttle_ms = max_throttle_ms
        self.min_wait_time = min_wait_time
        self.max_wait_time = max_wait_time
        self.last_rendering_time = None
        
        # Statistics tracking
        self.total_waits = 0
        self.total_wait_time = 0
        self.completion_detections = 0
        
        # Warm up the model
        self._warmup_model()
    
    def _warmup_model(self):
        """Warm up the model with a dummy image"""
        try:
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            self.classifier.predict(dummy_image)
            print("✅ Binary UI classifier warmed up")
        except Exception as e:
            print(f"⚠️  Model warmup failed: {e}")
    
    def wait_for_rendering(self, screenshot_provider: Callable, action_type: str = "general") -> float:
        """
        Wait until UI rendering is complete using AdaT approach
        
        Args:
            screenshot_provider: Function that returns screenshot (as np.array, PIL Image, or file path)
            action_type: Type of action that was performed ("click", "swipe", "type", etc.)
            
        Returns:
            The actual wait time in seconds
        """
        start_time = time.time()
        predict_tries = 0
        is_rendering_complete = False
        
        # Base wait times for different actions
        base_wait_times = {
            "click": 0.8,
            "tap": 0.8,
            "swipe": 1.5,
            "type": 1.2,
            "text": 1.2,
            "double_tap": 0.7,
            "long_press": 1.0,
            "general": 1.0
        }
        
        base_time = base_wait_times.get(action_type, 1.0)
        
        # If we don't have a working classifier, just use base time
        if not hasattr(self.classifier, 'model'):
            print("⚠️  No classifier available, using base wait time")
            time.sleep(base_time)
            self._update_stats(base_time, False)
            return base_time
        
        while (not is_rendering_complete and 
               predict_tries < self.max_model_tries and 
               (time.time() - start_time) * 1000 < self.max_throttle_ms):
            
            predict_tries += 1
            
            try:
                # Get current screenshot
                screenshot = screenshot_provider()
                if screenshot is None:
                    print("⚠️  Screenshot provider returned None")
                    continue
                
                # Predict if rendering is complete
                rendering_prob = self.classifier.predict(screenshot)
                is_rendering_complete = rendering_prob > 0.7  # Threshold for "rendered"
                
                print(f"🔍 UI completion check {predict_tries}: {rendering_prob:.3f} ({'✅ Complete' if is_rendering_complete else '⏳ Loading'})")
                
                if is_rendering_complete:
                    self.completion_detections += 1
                    break
                    
                # Wait a bit before checking again
                time.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️  Rendering check failed: {e}")
                break
        
        # Calculate actual wait time
        actual_wait = time.time() - start_time
        
        # Apply bounds
        bounded_wait = max(self.min_wait_time, min(actual_wait, self.max_wait_time))
        
        # If we didn't detect completion, use base time
        if not is_rendering_complete:
            bounded_wait = base_time
        
        print(f"⏳ UI rendering {'complete' if is_rendering_complete else 'timeout'}: waited {bounded_wait:.2f}s")
        
        self._update_stats(bounded_wait, is_rendering_complete)
        return bounded_wait
    
    def _update_stats(self, wait_time: float, completed: bool):
        """Update internal statistics"""
        self.total_waits += 1
        self.total_wait_time += wait_time
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about completion detection performance"""
        avg_wait = self.total_wait_time / max(1, self.total_waits)
        completion_rate = self.completion_detections / max(1, self.total_waits)
        
        return {
            "total_waits": self.total_waits,
            "total_wait_time": self.total_wait_time,
            "average_wait_time": avg_wait,
            "completion_detections": self.completion_detections,
            "completion_rate": completion_rate,
            "model_available": hasattr(self.classifier, 'model')
        }


class DynamicCompletionDetector:
    """
    High-level interface for dynamic completion detection
    Integrates with existing screenshot and action systems
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the completion detector
        
        Args:
            model_path: Path to the AdaT binary UI model (will download if None)
        """
        self.adaptive_waiter = AdaptiveWait(model_path)
        self.last_screenshot_path = None
        
    def wait_for_completion(self, 
                          screenshot_func: Callable, 
                          action_type: str = "general",
                          max_wait: float = 5.0) -> Dict[str, Any]:
        """
        Wait for UI action to complete dynamically
        
        Args:
            screenshot_func: Function that takes a screenshot and returns the path
            action_type: Type of action performed
            max_wait: Maximum time to wait
            
        Returns:
            Dictionary with completion info
        """
        def screenshot_provider():
            """Wrapper to provide screenshots to the adaptive waiter"""
            try:
                screenshot_path = screenshot_func()
                if screenshot_path and os.path.exists(screenshot_path):
                    self.last_screenshot_path = screenshot_path
                    return screenshot_path
                return None
            except Exception as e:
                print(f"⚠️  Screenshot provider error: {e}")
                return None
        
        start_time = time.time()
        
        # Set temporary max wait time
        original_max_wait = self.adaptive_waiter.max_wait_time
        self.adaptive_waiter.max_wait_time = max_wait
        
        try:
            actual_wait = self.adaptive_waiter.wait_for_rendering(
                screenshot_provider, 
                action_type
            )
            
            completion_time = time.time() - start_time
            
            return {
                "completed": True,
                "wait_time": actual_wait,
                "total_time": completion_time,
                "action_type": action_type,
                "last_screenshot": self.last_screenshot_path,
                "method": "adaptive_wait"
            }
            
        except Exception as e:
            print(f"❌ Completion detection failed: {e}")
            return {
                "completed": False,
                "wait_time": 0,
                "total_time": time.time() - start_time,
                "action_type": action_type,
                "error": str(e),
                "method": "fallback"
            }
        finally:
            # Restore original max wait time
            self.adaptive_waiter.max_wait_time = original_max_wait
    
    def is_ui_ready(self, screenshot_path: str) -> Dict[str, Any]:
        """
        Check if UI is ready/rendered for the next action
        
        Args:
            screenshot_path: Path to screenshot image
            
        Returns:
            Dictionary with readiness info
        """
        try:
            rendering_prob = self.adaptive_waiter.classifier.predict(screenshot_path)
            is_ready = rendering_prob > 0.7
            
            return {
                "ready": is_ready,
                "confidence": rendering_prob,
                "threshold": 0.7,
                "screenshot": screenshot_path
            }
            
        except Exception as e:
            print(f"❌ UI readiness check failed: {e}")
            return {
                "ready": True,  # Default to ready if check fails
                "confidence": 0.5,
                "error": str(e),
                "screenshot": screenshot_path
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get completion detection statistics"""
        return self.adaptive_waiter.get_statistics()


# Global instance for easy access
_global_completion_detector = None

def get_completion_detector() -> DynamicCompletionDetector:
    """Get or create the global completion detector instance"""
    global _global_completion_detector
    if _global_completion_detector is None:
        _global_completion_detector = DynamicCompletionDetector()
    return _global_completion_detector

def wait_for_action_completion(screenshot_func: Callable, 
                             action_type: str = "general",
                             max_wait: float = 5.0) -> Dict[str, Any]:
    """
    Convenience function for waiting for action completion
    
    Args:
        screenshot_func: Function that takes a screenshot
        action_type: Type of action performed
        max_wait: Maximum wait time
        
    Returns:
        Completion result dictionary
    """
    detector = get_completion_detector()
    return detector.wait_for_completion(screenshot_func, action_type, max_wait)

def check_ui_readiness(screenshot_path: str) -> bool:
    """
    Convenience function to check if UI is ready
    
    Args:
        screenshot_path: Path to screenshot
        
    Returns:
        True if UI is ready for next action
    """
    detector = get_completion_detector()
    result = detector.is_ui_ready(screenshot_path)
    return result.get("ready", True)


if __name__ == "__main__":
    # Test the completion detection system
    print("🧪 Testing Dynamic Completion Detection")
    print("=" * 50)
    
    # Initialize detector
    detector = DynamicCompletionDetector()
    
    # Test with a dummy screenshot function
    def dummy_screenshot():
        # Create a dummy image for testing
        dummy_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        test_path = "test_screenshot.png"
        cv2.imwrite(test_path, dummy_img)
        return test_path
    
    # Test completion waiting
    print("🔍 Testing completion detection...")
    result = detector.wait_for_completion(dummy_screenshot, "click", 2.0)
    print(f"Result: {result}")
    
    # Test UI readiness
    if os.path.exists("test_screenshot.png"):
        print("🔍 Testing UI readiness...")
        readiness = detector.is_ui_ready("test_screenshot.png")
        print(f"UI Readiness: {readiness}")
        
        # Clean up
        os.remove("test_screenshot.png")
    
    # Show statistics
    stats = detector.get_stats()
    print(f"📊 Statistics: {stats}")
    
    print("✅ Testing completed!")
