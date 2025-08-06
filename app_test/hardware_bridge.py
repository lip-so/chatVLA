#!/usr/bin/env python3
"""
Comprehensive hardware communication bridge for koch robot
Enables full cloud-to-hardware communication
"""

import serial
import serial.tools.list_ports
import json
import time
from pathlib import Path

class ComprehensiveHardwareBridge:
    def __init__(self, robot_type="koch"):
        self.robot_type = robot_type
        self.connection = None
        self.config = {'dof': 6, 'type': 'leader_follower', 'description': 'Koch Follower (6-DOF leader-follower arm)', 'baud_rate': 1000000, 'port_pattern': '/dev/ttyUSB*'}
        
    def scan_ports(self):
        """Scan for available serial ports"""
        ports = []
        try:
            for port in serial.tools.list_ports.comports():
                ports.append({
                    'device': port.device,
                    'description': port.description,
                    'hwid': port.hwid,
                    'compatible': self._check_compatibility(port)
                })
        except Exception as e:
            print(f"Port scanning error: {e}")
        return ports
    
    def _check_compatibility(self, port):
        """Check if port is compatible with robot"""
        # Basic compatibility check
        usb_keywords = ['USB', 'ACM', 'ttyUSB', 'ttyACM']
        return any(keyword in port.device for keyword in usb_keywords)
    
    def connect(self, port, baud_rate=None):
        """Connect to robot hardware"""
        try:
            baud_rate = baud_rate or self.config['baud_rate']
            self.connection = serial.Serial(port, baud_rate, timeout=1.0)
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from robot hardware"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def send_command(self, command):
        """Send command to robot"""
        if self.connection:
            try:
                self.connection.write(command.encode())
                return self.connection.readline().decode().strip()
            except Exception as e:
                print(f"Command failed: {e}")
                return None
        return None

if __name__ == "__main__":
    bridge = ComprehensiveHardwareBridge()
    ports = bridge.scan_ports()
    print(f"Available ports: {json.dumps(ports, indent=2)}")
