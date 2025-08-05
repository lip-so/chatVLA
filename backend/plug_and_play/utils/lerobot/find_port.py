#!/usr/bin/env python3
"""
LeRobot USB Port Detection Tool

This script helps identify the USB ports associated with leader and follower robotic arms
through an interactive process that guides users step-by-step.
"""

import serial
import serial.tools.list_ports
import time
import os
import sys
from typing import List, Dict, Optional, Set


class USBPortDetector:
    """
    Interactive USB port detection for robotic arms.
    Provides step-by-step guidance to identify leader and follower arm ports.
    """
    
    def __init__(self):
        self.initial_ports: Set[str] = set()
        self.leader_port: Optional[str] = None
        self.follower_port: Optional[str] = None
        
    def get_available_ports(self) -> List[Dict[str, str]]:
        """Get list of available serial ports with details."""
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append({
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid,
                'vid': getattr(port, 'vid', None),
                'pid': getattr(port, 'pid', None),
                'serial_number': getattr(port, 'serial_number', None)
            })
        return ports
    
    def display_ports(self, ports: List[Dict[str, str]], title: str = "Available Ports"):
        """Display formatted list of ports."""
        print(f"\n{title}")
        print("=" * 60)
        
        if not ports:
            print("No serial ports detected.")
            return
            
        for i, port in enumerate(ports, 1):
            print(f"{i}. {port['device']}")
            print(f"   Description: {port['description']}")
            if port['vid'] and port['pid']:
                print(f"   VID:PID: {port['vid']:04X}:{port['pid']:04X}")
            if port['serial_number']:
                print(f"   Serial: {port['serial_number']}")
            print(f"   Hardware ID: {port['hwid']}")
            print()
    
    def wait_for_user_input(self, message: str) -> None:
        """Display message and wait for user to press Enter."""
        print(f"\n{message}")
        print("Press ENTER when ready...")
        input()
    
    def detect_port_changes(self, initial_ports: Set[str]) -> Optional[str]:
        """Detect which port was added or removed."""
        current_ports = {port['device'] for port in self.get_available_ports()}
        
        # Check for newly connected ports
        new_ports = current_ports - initial_ports
        if new_ports:
            return list(new_ports)[0]
            
        return None
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print application header."""
        self.clear_screen()
        print("=" * 70)
        print("                LeRobot USB Port Detection Tool")
        print("=" * 70)
        print()
        print("This tool will help you identify the USB ports for your robotic arms.")
        print("Please follow the instructions carefully for accurate detection.")
        print()
    
    def detect_leader_arm_port(self) -> Optional[str]:
        """Interactive process to detect leader arm port."""
        print("STEP 1: Detecting Leader Arm Port")
        print("-" * 40)
        print()
        
        # Show initial ports
        initial_ports = self.get_available_ports()
        self.display_ports(initial_ports, "Current Connected Ports")
        
        # Store initial port names
        self.initial_ports = {port['device'] for port in initial_ports}
        
        # Instruct user to unplug leader arm
        self.wait_for_user_input(
            "📋 INSTRUCTION:\n"
            "1. Locate the USB cable connected to your LEADER ARM\n"
            "2. UNPLUG the USB cable from the leader arm\n"
            "3. Wait for the disconnection to register"
        )
        
        # Check for removed ports
        time.sleep(2)  # Allow time for system to register disconnection
        
        current_ports = self.get_available_ports()
        current_port_names = {port['device'] for port in current_ports}
        
        disconnected_ports = self.initial_ports - current_port_names
        
        if disconnected_ports:
            suspected_leader_port = list(disconnected_ports)[0]
            print(f"Detected disconnection: {suspected_leader_port}")
        else:
            print("WARNING: No port disconnection detected. Please try again.")
            return None
        
        # Instruct user to reconnect leader arm
        self.wait_for_user_input(
            "📋 CONFIRMATION STEP:\n"
            "1. RECONNECT the USB cable to your LEADER ARM\n"
            "2. Wait for the connection to register"
        )
        
        # Verify reconnection
        time.sleep(2)
        final_ports = self.get_available_ports()
        final_port_names = {port['device'] for port in final_ports}
        
        if suspected_leader_port in final_port_names:
            print(f"LEADER ARM PORT CONFIRMED: {suspected_leader_port}")
            return suspected_leader_port
        else:
            print("WARNING: Port confirmation failed. Please try the process again.")
            return None
    
    def detect_follower_arm_port(self) -> Optional[str]:
        """Interactive process to detect follower arm port."""
        print("\nSTEP 2: Detecting Follower Arm Port")
        print("-" * 40)
        print()
        
        # Show current ports
        current_ports = self.get_available_ports()
        self.display_ports(current_ports, "Current Connected Ports")
        
        current_port_names = {port['device'] for port in current_ports}
        
        # Instruct user to unplug follower arm
        self.wait_for_user_input(
            "📋 INSTRUCTION:\n"
            "1. Locate the USB cable connected to your FOLLOWER ARM\n"
            "2. UNPLUG the USB cable from the follower arm\n"
            "3. Wait for the disconnection to register"
        )
        
        # Check for removed ports
        time.sleep(2)
        
        updated_ports = self.get_available_ports()
        updated_port_names = {port['device'] for port in updated_ports}
        
        disconnected_ports = current_port_names - updated_port_names
        
        if disconnected_ports:
            suspected_follower_port = list(disconnected_ports)[0]
            print(f"Detected disconnection: {suspected_follower_port}")
        else:
            print("WARNING: No port disconnection detected. Please try again.")
            return None
        
        # Instruct user to reconnect follower arm
        self.wait_for_user_input(
            "📋 CONFIRMATION STEP:\n"
            "1. RECONNECT the USB cable to your FOLLOWER ARM\n"
            "2. Wait for the connection to register"
        )
        
        # Verify reconnection
        time.sleep(2)
        final_ports = self.get_available_ports()
        final_port_names = {port['device'] for port in final_ports}
        
        if suspected_follower_port in final_port_names:
            print(f"FOLLOWER ARM PORT CONFIRMED: {suspected_follower_port}")
            return suspected_follower_port
        else:
            print("WARNING: Port confirmation failed. Please try the process again.")
            return None
    
    def save_port_configuration(self, leader_port: str, follower_port: str) -> None:
        """Save detected port configuration to a file."""
        config_content = f"""# LeRobot Port Configuration
# Generated by USB Port Detection Tool

LEADER_ARM_PORT = "{leader_port}"
FOLLOWER_ARM_PORT = "{follower_port}"

# Port Details:
# Leader Arm:  {leader_port}
# Follower Arm: {follower_port}
"""
        
        config_file = "lerobot_ports.py"
        
        try:
            with open(config_file, 'w') as f:
                f.write(config_content)
            print(f"Port configuration saved to: {config_file}")
        except Exception as e:
            print(f"WARNING: Failed to save configuration: {e}")
    
    def display_final_results(self, leader_port: str, follower_port: str) -> None:
        """Display final detection results."""
        print("\n" + "=" * 70)
        print("                    PORT DETECTION COMPLETE")
        print("=" * 70)
        print()
        print("SUCCESS! Robotic arm ports have been identified:")
        print()
        print(f"🦾 LEADER ARM PORT:   {leader_port}")
        print(f"🤖 FOLLOWER ARM PORT: {follower_port}")
        print()
        print("Configuration has been saved to 'lerobot_ports.py'")
        print()
        print("You can now use these ports in your LeRobot applications:")
        print(f"   - Leader:   serial.Serial('{leader_port}')")
        print(f"   - Follower: serial.Serial('{follower_port}')")
        print()
        print("=" * 70)
    
    def run_detection(self) -> bool:
        """Run the complete port detection process."""
        try:
            self.print_header()
            
            # Check if pyserial is available
            try:
                import serial
                import serial.tools.list_ports
            except ImportError:
                print("ERROR: pyserial library is not installed.")
                print("Please install it using: pip install pyserial")
                return False
            
            # Check if any serial ports are available
            initial_ports = self.get_available_ports()
            if not initial_ports:
                print("ERROR: No serial ports detected.")
                print("Please ensure your robotic arms are connected via USB.")
                return False
            
            print("🚀 Starting USB port detection process...")
            print("Make sure both robotic arms are connected before proceeding.")
            print()
            
            self.wait_for_user_input("Press ENTER to begin detection process")
            
            # Detect leader arm port
            leader_port = self.detect_leader_arm_port()
            if not leader_port:
                print("❌ Failed to detect leader arm port. Please try again.")
                return False
            
            # Detect follower arm port
            follower_port = self.detect_follower_arm_port()
            if not follower_port:
                print("❌ Failed to detect follower arm port. Please try again.")
                return False
            
            # Verify ports are different
            if leader_port == follower_port:
                print("❌ ERROR: Leader and follower arms detected on same port.")
                print("This likely indicates a detection error. Please retry.")
                return False
            
            # Save configuration and display results
            self.save_port_configuration(leader_port, follower_port)
            self.display_final_results(leader_port, follower_port)
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Detection process cancelled by user.")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error during detection: {e}")
            return False


def main():
    """Main entry point for the port detection tool."""
    detector = USBPortDetector()
    
    try:
        success = detector.run_detection()
        if success:
            print("\nPort detection completed successfully!")
        else:
            print("\nPort detection failed. Please check the instructions and try again.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()