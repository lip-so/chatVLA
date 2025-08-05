#!/usr/bin/env python3
"""
Demo of dynamic port detection similar to ref/lerobot/find_port.py
This shows how the plug/unplug detection works.
"""

import time
import os
from pathlib import Path

try:
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("WARNING: pyserial not available - using simulation mode")

def get_available_ports():
    """Get all available serial ports"""
    ports = []
    
    if SERIAL_AVAILABLE:
        for port in serial.tools.list_ports.comports():
            ports.append({
                'device': port.device,
                'description': port.description,
                'manufacturer': getattr(port, 'manufacturer', 'Unknown'),
                'product': getattr(port, 'product', 'Unknown')
            })
    else:
        # Simulate some ports for demo
        import platform
        if platform.system() == "Darwin":  # macOS
            ports = [
                {'device': '/dev/cu.debug-console', 'description': 'Debug Console'},
                {'device': '/dev/cu.Bluetooth-Incoming-Port', 'description': 'Bluetooth'}
            ]
        else:
            ports = [
                {'device': '/dev/ttyS0', 'description': 'Serial Port 0'},
                {'device': '/dev/ttyS1', 'description': 'Serial Port 1'}
            ]
    
    return ports

def demo_port_detection():
    """Demo the interactive port detection process"""
    print("🤖 LeRobot Dynamic Port Detection Demo")
    print("=" * 50)
    print("This demonstrates the plug/unplug detection system")
    print("(similar to ref/lerobot/find_port.py)\n")
    
    # Step 1: Get baseline ports
    print("📍 Step 1: Establishing baseline...")
    baseline_ports = get_available_ports()
    baseline_devices = {p['device'] for p in baseline_ports}
    
    print(f"Found {len(baseline_ports)} baseline ports:")
    for port in baseline_ports:
        print(f"  • {port['device']} - {port['description']}")
    
    print("\nStep 2: Monitoring for changes...")
    print("Connect/disconnect your robot arms to see real-time detection!")
    print("Press Ctrl+C to stop monitoring\n")
    
    detected_robot_ports = {}
    
    try:
        while True:
            current_ports = get_available_ports()
            current_devices = {p['device'] for p in current_ports}
            
            # Find newly connected ports
            new_ports = current_devices - baseline_devices
            # Find disconnected ports
            removed_ports = baseline_devices - current_devices
            
            if new_ports:
                print(f" NEW PORT DETECTED: {list(new_ports)}")
                for port_device in new_ports:
                    port_info = next((p for p in current_ports if p['device'] == port_device), None)
                    if port_info:
                        print(f"   Device: {port_info['device']}")
                        print(f"   Description: {port_info['description']}")
                        
                        # Simulate assignment
                        if 'leader' not in detected_robot_ports:
                            detected_robot_ports['leader'] = port_device
                            print(f"   🦾 Assigned to LEADER arm")
                        elif 'follower' not in detected_robot_ports:
                            detected_robot_ports['follower'] = port_device
                            print(f"   🤖 Assigned to FOLLOWER arm")
                        
                        print()
                
                # Update baseline to include new ports
                baseline_devices.update(new_ports)
            
            if removed_ports:
                print(f"PORT DISCONNECTED: {list(removed_ports)}")
                # Remove from assignments if they were assigned
                for arm_type, port in list(detected_robot_ports.items()):
                    if port in removed_ports:
                        del detected_robot_ports[arm_type]
                        print(f"   🚫 Removed {arm_type} assignment")
                
                # Update baseline
                baseline_devices -= removed_ports
                print()
            
            # Show current assignments
            if detected_robot_ports:
                print("\r📋 Current assignments:", end="")
                for arm_type, port in detected_robot_ports.items():
                    print(f" {arm_type.upper()}={port}", end="")
                print(" " * 20, end="")  # Clear line
            else:
                print("\rWaiting for robot connections..." + " " * 30, end="")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped")
        
        if detected_robot_ports:
            print("\n📋 Final Port Assignments:")
            print("-" * 30)
            for arm_type, port in detected_robot_ports.items():
                print(f"  {arm_type.upper()}: {port}")
            
            # Simulate saving to config
            print(f"\nConfiguration would be saved to robot_config.json:")
            print(f"{{")
            for arm_type, port in detected_robot_ports.items():
                print(f'  "{arm_type}_port": "{port}",')
            print(f'  "robot_type": "your_robot"')
            print(f"}}")
        else:
            print("\nERROR: No robot ports were detected")
        
        print("\nDemo complete!")

if __name__ == "__main__":
    demo_port_detection()