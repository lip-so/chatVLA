#!/usr/bin/env python3

"""
USB Port Detection for Robot Connections
Provides cross-platform USB port detection and identification
"""

import sys
import logging
from typing import List, Dict, Any
from pathlib import Path

# Add ref path for LeRobot imports
ref_path = Path(__file__).parent.parent.parent / "ref"
sys.path.insert(0, str(ref_path))

logger = logging.getLogger(__name__)

# Try to import serial for port detection
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logger.warning("pyserial not available - port detection will be limited")

class USBPortDetector:
    """Cross-platform USB port detection"""
    
    def __init__(self):
        self.serial_available = SERIAL_AVAILABLE
    
    def scan_ports(self) -> List[Dict[str, Any]]:
        """Scan for available USB ports"""
        ports = []
        
        if not self.serial_available:
            # Return mock ports for development/demo
            return self._get_mock_ports()
        
        try:
            # Use pyserial to detect ports
            available_ports = serial.tools.list_ports.comports()
            
            for port in available_ports:
                port_info = {
                    'port': port.device,
                    'description': port.description or 'Unknown Device',
                    'manufacturer': getattr(port, 'manufacturer', None) or 'Unknown',
                    'product': getattr(port, 'product', None),
                    'serial_number': getattr(port, 'serial_number', None),
                    'vid': getattr(port, 'vid', None),
                    'pid': getattr(port, 'pid', None),
                    'location': getattr(port, 'location', None),
                }
                
                # Add robot-specific hints
                port_info['robot_hints'] = self._get_robot_hints(port_info)
                
                ports.append(port_info)
            
            # Sort by port name for consistency
            ports.sort(key=lambda x: x['port'])
            
            logger.info(f"Found {len(ports)} USB ports")
            return ports
            
        except Exception as e:
            logger.error(f"Port scanning failed: {e}")
            # Return mock ports as fallback
            return self._get_mock_ports()
    
    def _get_robot_hints(self, port_info: Dict[str, Any]) -> List[str]:
        """Get hints about what robot types might work with this port"""
        hints = []
        
        description = port_info.get('description', '').lower()
        manufacturer = port_info.get('manufacturer', '').lower()
        
        # Common robot controller identifiers
        if 'arduino' in description or 'arduino' in manufacturer:
            hints.extend(['SO101', 'SO100', 'Koch'])
        elif 'ftdi' in description or 'ftdi' in manufacturer:
            hints.extend(['SO101', 'SO100', 'ViperX'])
        elif 'ch340' in description or 'ch341' in description:
            hints.extend(['SO101', 'SO100'])
        elif 'cp210' in description or 'cp2102' in description:
            hints.extend(['Various robots'])
        elif 'usb serial' in description:
            hints.extend(['Generic robot controller'])
        
        # VID/PID specific hints
        vid = port_info.get('vid')
        if vid:
            if vid == 0x2341:  # Arduino
                hints.extend(['Arduino-compatible robots'])
            elif vid == 0x0403:  # FTDI
                hints.extend(['FTDI-based controllers'])
            elif vid == 0x1a86:  # CH340/CH341
                hints.extend(['CH340-based controllers'])
        
        return list(set(hints))  # Remove duplicates
    
    def _get_mock_ports(self) -> List[Dict[str, Any]]:
        """Return mock ports for development/demo purposes"""
        import platform
        
        system = platform.system().lower()
        
        if system == 'darwin':  # macOS
            return [
                {
                    'port': '/dev/cu.usbmodem585A0076891',
                    'description': 'Arduino Uno',
                    'manufacturer': 'Arduino LLC',
                    'product': 'Arduino Uno',
                    'serial_number': '585A0076891',
                    'vid': 0x2341,
                    'pid': 0x0043,
                    'location': '20-1',
                    'robot_hints': ['SO101', 'SO100', 'Koch']
                },
                {
                    'port': '/dev/cu.usbmodem58760431551',
                    'description': 'Arduino Mega 2560',
                    'manufacturer': 'Arduino LLC',
                    'product': 'Arduino Mega 2560',
                    'serial_number': '58760431551',
                    'vid': 0x2341,
                    'pid': 0x0042,
                    'location': '20-2',
                    'robot_hints': ['SO101', 'SO100']
                },
                {
                    'port': '/dev/cu.usbserial-FT6S4DSA',
                    'description': 'FT232R USB UART',
                    'manufacturer': 'FTDI',
                    'product': 'FT232R USB UART',
                    'serial_number': 'FT6S4DSA',
                    'vid': 0x0403,
                    'pid': 0x6001,
                    'location': '20-3',
                    'robot_hints': ['ViperX', 'SO101']
                }
            ]
        elif system == 'linux':
            return [
                {
                    'port': '/dev/ttyUSB0',
                    'description': 'FT232R USB UART',
                    'manufacturer': 'FTDI',
                    'product': 'FT232R USB UART',
                    'serial_number': 'FT6S4DSA',
                    'vid': 0x0403,
                    'pid': 0x6001,
                    'location': '1-1.2:1.0',
                    'robot_hints': ['SO101', 'SO100', 'ViperX']
                },
                {
                    'port': '/dev/ttyUSB1',
                    'description': 'FT232R USB UART',
                    'manufacturer': 'FTDI',
                    'product': 'FT232R USB UART',
                    'serial_number': 'FT6S4DSB',
                    'vid': 0x0403,
                    'pid': 0x6001,
                    'location': '1-1.3:1.0',
                    'robot_hints': ['SO101', 'SO100', 'ViperX']
                },
                {
                    'port': '/dev/ttyACM0',
                    'description': 'Arduino Uno',
                    'manufacturer': 'Arduino LLC',
                    'product': 'Arduino Uno',
                    'serial_number': '585A0076891',
                    'vid': 0x2341,
                    'pid': 0x0043,
                    'location': '1-1.4:1.0',
                    'robot_hints': ['Koch', 'SO101']
                }
            ]
        else:  # Windows
            return [
                {
                    'port': 'COM3',
                    'description': 'Arduino Uno (COM3)',
                    'manufacturer': 'Arduino LLC',
                    'product': 'Arduino Uno',
                    'serial_number': '585A0076891',
                    'vid': 0x2341,
                    'pid': 0x0043,
                    'location': 'Port_#0003.Hub_#0001',
                    'robot_hints': ['SO101', 'SO100', 'Koch']
                },
                {
                    'port': 'COM4',
                    'description': 'USB-SERIAL CH340 (COM4)',
                    'manufacturer': 'wch.cn',
                    'product': 'USB-SERIAL CH340',
                    'serial_number': None,
                    'vid': 0x1a86,
                    'pid': 0x7523,
                    'location': 'Port_#0004.Hub_#0001',
                    'robot_hints': ['SO101', 'SO100']
                },
                {
                    'port': 'COM5',
                    'description': 'FT232R USB UART (COM5)',
                    'manufacturer': 'FTDI',
                    'product': 'FT232R USB UART',
                    'serial_number': 'FT6S4DSA',
                    'vid': 0x0403,
                    'pid': 0x6001,
                    'location': 'Port_#0005.Hub_#0001',
                    'robot_hints': ['ViperX', 'SO101']
                }
            ]
    
    def test_port_connection(self, port: str, baudrate: int = 115200, timeout: float = 1.0) -> bool:
        """Test if a port can be opened for communication"""
        if not self.serial_available:
            logger.warning("Cannot test port connection - pyserial not available")
            return False
        
        try:
            with serial.Serial(port, baudrate, timeout=timeout) as ser:
                return ser.is_open
        except Exception as e:
            logger.debug(f"Port {port} test failed: {e}")
            return False
    
    def get_port_info(self, port_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific port"""
        ports = self.scan_ports()
        for port in ports:
            if port['port'] == port_name:
                return port
        
        return {
            'port': port_name,
            'description': 'Unknown Device',
            'manufacturer': 'Unknown',
            'robot_hints': []
        }

def detect_cameras() -> List[Dict[str, Any]]:
    """Detect available cameras"""
    cameras = []
    
    try:
        # Try to import OpenCV for camera detection
        import cv2
        
        # Test camera indices 0-5
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                cameras.append({
                    'index': i,
                    'name': f'Camera {i}',
                    'width': width if width > 0 else 640,
                    'height': height if height > 0 else 480,
                    'fps': fps if fps > 0 else 30,
                    'backend': 'OpenCV'
                })
                
                cap.release()
        
        logger.info(f"Found {len(cameras)} cameras")
        
    except ImportError:
        logger.warning("OpenCV not available - camera detection disabled")
        # Return mock cameras for demo
        cameras = [
            {
                'index': 0,
                'name': 'Built-in Camera',
                'width': 1280,
                'height': 720,
                'fps': 30,
                'backend': 'Mock'
            }
        ]
    except Exception as e:
        logger.error(f"Camera detection failed: {e}")
    
    return cameras

# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    detector = USBPortDetector()
    ports = detector.scan_ports()
    
    print(f"Found {len(ports)} USB ports:")
    for port in ports:
        print(f"  {port['port']}: {port['description']} ({port['manufacturer']})")
        if port['robot_hints']:
            print(f"    Robot hints: {', '.join(port['robot_hints'])}")
    
    cameras = detect_cameras()
    print(f"\nFound {len(cameras)} cameras:")
    for camera in cameras:
        print(f"  Camera {camera['index']}: {camera['width']}x{camera['height']} @ {camera['fps']}fps")
