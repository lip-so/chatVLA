#!/usr/bin/env python3
"""
Hardware communication bridge for so101 robot
Enables cloud server to communicate with local hardware
"""

import serial
import serial.tools.list_ports
import json
import time
from pathlib import Path

class HardwareBridge:
    def __init__(self, robot_type="so101"):
        self.robot_type = robot_type
        self.connection = None
        
    def scan_ports(self):
        """Scan for available serial ports"""
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append({
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid
            })
        return ports
    
    def connect(self, port, baud_rate=1000000):
        """Connect to robot hardware"""
        try:
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

if __name__ == "__main__":
    bridge = HardwareBridge()
    ports = bridge.scan_ports()
    print(f"Available ports: {json.dumps(ports, indent=2)}")
