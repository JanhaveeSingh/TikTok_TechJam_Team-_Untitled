#!/usr/bin/env python3
"""
Unified Parser Interface
Provides a single interface for both OmniParser and CLIP Parser
"""

import requests
import time
import json
import tempfile
import os
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import base64
from io import BytesIO

from parser_config import get_parser_config, get_current_parser

class UnifiedParser:
    """Unified interface for different UI parsers"""
    
    def __init__(self):
        self.config = get_parser_config()
        self._last_parser_used = None
        
    def parse_image(self, 
                   image_path: str, 
                   box_threshold: float = 0.3,
                   iou_threshold: float = 0.5,
                   imgsz: int = 640,
                   force_parser: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse UI elements from an image using the active parser
        
        Args:
            image_path: Path to the image file
            box_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
            imgsz: Image size for processing
            force_parser: Force use of specific parser ("omni" or "clip")
            
        Returns:
            Parsed results in standardized format
        """
        # Determine which parser to use
        if force_parser:
            if force_parser == "omni":
                parser_uri = self.config.omni_uri
                parser_type = "omni"
            elif force_parser == "clip":
                parser_uri = self.config.clip_uri
                parser_type = "clip"
            else:
                raise ValueError(f"Invalid parser type: {force_parser}")
        else:
            parser_type, parser_uri = get_current_parser()
        
        self._last_parser_used = parser_type
        
        try:
            if parser_type == "omni":
                return self._parse_with_omni(image_path, parser_uri, box_threshold, iou_threshold, imgsz)
            elif parser_type == "clip":
                return self._parse_with_clip(image_path, parser_uri, box_threshold, iou_threshold, imgsz)
            else:
                raise ValueError(f"Unknown parser type: {parser_type}")
                
        except Exception as e:
            print(f"❌ Error with {parser_type} parser: {e}")
            
            # Try fallback parser if not forced
            if not force_parser and parser_type != self.config.fallback_parser:
                print(f"🔄 Trying fallback parser: {self.config.fallback_parser}")
                return self.parse_image(image_path, box_threshold, iou_threshold, imgsz, 
                                      force_parser=self.config.fallback_parser)
            else:
                raise
    
    def _parse_with_omni(self, image_path: str, parser_uri: str, 
                        box_threshold: float, iou_threshold: float, imgsz: int) -> Dict[str, Any]:
        """Parse image using OmniParser"""
        start_time = time.time()
        
        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}
            data = {
                'box_threshold': box_threshold,
                'iou_threshold': iou_threshold,
                'imgsz': imgsz
            }
            
            response = requests.post(
                f"{parser_uri}/process_image/",
                files=files,
                data=data,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            result = response.json()
        
        # Standardize the result format
        return self._standardize_omni_result(result, time.time() - start_time)
    
    def _parse_with_clip(self, image_path: str, parser_uri: str,
                        box_threshold: float, iou_threshold: float, imgsz: int) -> Dict[str, Any]:
        """Parse image using CLIP Parser"""
        start_time = time.time()
        
        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}
            data = {
                'box_threshold': box_threshold,
                'iou_threshold': iou_threshold,
                'imgsz_component': imgsz  # CLIP parser uses different parameter name
            }
            
            response = requests.post(
                f"{parser_uri}/process_image/",
                files=files,
                data=data,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            result = response.json()
        
        # Standardize the result format  
        return self._standardize_clip_result(result, time.time() - start_time)
    
    def _standardize_omni_result(self, result: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
        """Standardize OmniParser result format"""
        parsed_content = result.get('parsed_content', [])
        
        # Ensure all elements have required fields
        for item in parsed_content:
            if 'ui_type' not in item:
                item['ui_type'] = 'unknown'
            if 'confidence' not in item:
                item['confidence'] = 1.0
        
        return {
            'status': 'success',
            'parser_used': 'omni',
            'processing_time': processing_time,
            'parsed_content': parsed_content,
            'labeled_image': result.get('labeled_image', ''),
            'element_count': len(parsed_content),
            'metadata': {
                'parser_type': 'omniparser',
                'has_text_elements': any(item.get('type') == 'text' for item in parsed_content),
                'has_ui_elements': any(item.get('type') != 'text' for item in parsed_content)
            }
        }
    
    def _standardize_clip_result(self, result: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
        """Standardize CLIP Parser result format"""
        parsed_content = result.get('parsed_content', [])
        
        # CLIP parser already has ui_type and confidence fields
        return {
            'status': 'success',
            'parser_used': 'clip',
            'processing_time': processing_time,
            'parsed_content': parsed_content,
            'labeled_image': result.get('labeled_image', ''),
            'element_count': len(parsed_content),
            'metadata': {
                'parser_type': 'clip',
                'has_text_elements': any(item.get('type') == 'text' for item in parsed_content),
                'has_ui_elements': any(item.get('type') != 'text' for item in parsed_content),
                'clip_classification': True
            }
        }
    
    def get_parser_status(self) -> Dict[str, Any]:
        """Get current parser status and configuration"""
        return {
            'current_parser': self._last_parser_used,
            'config': self.config.get_parser_info(),
            'available_parsers': self.config.get_available_parsers()
        }
    
    def benchmark_parsers(self, image_path: str, iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark both parsers on the same image
        
        Args:
            image_path: Path to test image
            iterations: Number of test iterations
            
        Returns:
            Benchmark results
        """
        results = {
            'omni': {'times': [], 'success': 0, 'errors': []},
            'clip': {'times': [], 'success': 0, 'errors': []}
        }
        
        for parser_type in ['omni', 'clip']:
            for i in range(iterations):
                try:
                    start_time = time.time()
                    result = self.parse_image(image_path, force_parser=parser_type)
                    end_time = time.time()
                    
                    results[parser_type]['times'].append(end_time - start_time)
                    results[parser_type]['success'] += 1
                    
                except Exception as e:
                    results[parser_type]['errors'].append(str(e))
        
        # Calculate statistics
        for parser_type in results:
            times = results[parser_type]['times']
            if times:
                results[parser_type]['avg_time'] = sum(times) / len(times)
                results[parser_type]['min_time'] = min(times)
                results[parser_type]['max_time'] = max(times)
            else:
                results[parser_type]['avg_time'] = None
        
        return results

# Global unified parser instance
unified_parser = UnifiedParser()

def parse_ui_elements(image_path: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to parse UI elements
    Compatible with existing code
    """
    return unified_parser.parse_image(image_path, **kwargs)

def get_parser_status() -> Dict[str, Any]:
    """Get current parser status"""
    return unified_parser.get_parser_status()

if __name__ == "__main__":
    # Test the unified parser
    print("🧪 Testing Unified Parser")
    print("=" * 50)
    
    # Show parser status
    status = get_parser_status()
    print(f"Parser Status: {json.dumps(status, indent=2)}")
    
    # Test with a sample image if available
    test_image = "log/screenshots/test/test_step1_20250830_184318.png"
    if os.path.exists(test_image):
        print(f"\n🖼️ Testing with image: {test_image}")
        try:
            result = parse_ui_elements(test_image)
            print(f"✅ Parsing successful with {result['parser_used']} parser")
            print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
            print(f"📊 Elements found: {result['element_count']}")
        except Exception as e:
            print(f"❌ Parsing failed: {e}")
    else:
        print(f"⚠️ Test image not found: {test_image}")
