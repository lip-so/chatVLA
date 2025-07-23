"""
Backend service for LeRobot installation operations.
Handles all installation logic and system interactions.
"""

import subprocess
import shutil
import os
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
from enum import Enum
import time

from error_handler import ErrorHandler, InstallationError

try:
    import serial
    import serial.tools.list_ports
    PYSERIAL_AVAILABLE = True
except ImportError:
    PYSERIAL_AVAILABLE = False


class InstallationStep(Enum):
    CHECKING_PREREQUISITES = "checking_prerequisites"
    CLONING_REPOSITORY = "cloning_repository"
    CREATING_ENVIRONMENT = "creating_environment"
    INSTALLING_FFMPEG = "installing_ffmpeg"
    INSTALLING_LEROBOT = "installing_lerobot"
    DETECTING_USB_PORTS = "detecting_usb_ports"
    COMPLETED = "completed"
    FAILED = "failed"


class InstallationResult:
    def __init__(self, success: bool, message: str, details: str = ""):
        self.success = success
        self.message = message
        self.details = details


class InstallationService:
    """
    Service class that handles all LeRobot installation operations.
    Provides callbacks for UI updates and progress tracking.
    """
    
    def __init__(self):
        self.is_running = False
        self.current_step = None
        self.installation_path = None
        
        # Callbacks for UI updates
        self.on_progress_update: Optional[Callable[[InstallationStep, str], None]] = None
        self.on_log_message: Optional[Callable[[str], None]] = None
        self.on_completion: Optional[Callable[[InstallationResult], None]] = None
        
        # Error handler for consistent error processing
        self.error_handler = ErrorHandler(log_callback=self._log)
        
    def set_callbacks(self, 
                     progress_callback: Callable[[InstallationStep, str], None],
                     log_callback: Callable[[str], None],
                     completion_callback: Callable[[InstallationResult], None]):
        """Set callback functions for UI updates."""
        self.on_progress_update = progress_callback
        self.on_log_message = log_callback
        self.on_completion = completion_callback
        
    def start_installation(self, installation_path: str) -> bool:
        """Start the installation process."""
        if self.is_running:
            return False
            
        self.is_running = True
        self.installation_path = Path(installation_path)
        
        try:
            self._run_installation_steps()
            return True
        except Exception as e:
            # Use error handler for consistent error processing
            installation_error = self.error_handler.handle_error(e, "installation process")
            error_message = self.error_handler.get_user_friendly_message(installation_error)
            
            self._complete_installation(InstallationResult(
                False, installation_error.message, error_message
            ))
            return False
        finally:
            self.is_running = False
            
    def cancel_installation(self):
        """Cancel the running installation."""
        self.is_running = False
        
    def _run_installation_steps(self):
        """Execute all installation steps in sequence."""
        steps = [
            (InstallationStep.CHECKING_PREREQUISITES, self._check_prerequisites),
            (InstallationStep.CLONING_REPOSITORY, self._clone_repository),
            (InstallationStep.CREATING_ENVIRONMENT, self._create_environment),
            (InstallationStep.INSTALLING_FFMPEG, self._install_ffmpeg),
            (InstallationStep.INSTALLING_LEROBOT, self._install_lerobot),
            (InstallationStep.DETECTING_USB_PORTS, self._detect_usb_ports)
        ]
        
        for step, step_function in steps:
            if not self.is_running:
                self._complete_installation(InstallationResult(
                    False, "Installation cancelled"
                ))
                return
                
            self.current_step = step
            self._update_progress(step, f"Starting {step.value.replace('_', ' ')}")
            
            if not step_function():
                self._complete_installation(InstallationResult(
                    False, f"Installation failed at step: {step.value}"
                ))
                return
                
        # Add final USB port detection instructions
        self._log("\n" + "="*60)
        self._log("🎉 LeRobot Installation Complete!")
        self._log("="*60)
        self._log("")
        self._log("🔌 NEXT STEPS - USB Port Setup:")
        self._log("1. Connect your robotic arms via USB")
        self._log("2. Activate the environment: conda activate lerobot")  
        self._log("3. Navigate to your LeRobot directory")
        self._log("4. Run port detection: python lerobot/find_port.py")
        self._log("")
        self._log("💡 Alternative methods:")
        self._log("• Use the web interface for guided setup")
        self._log("• Run: python post_install_setup.py for status check")
        self._log("• Run: python activate_lerobot.py for activation help")
        self._log("")
        self._log("📖 Documentation and guides are available in your installation directory")
        
        self._complete_installation(InstallationResult(
            True, "Installation and USB port setup completed successfully"
        ))
        
    def _check_prerequisites(self) -> bool:
        """Check if required tools are available."""
        self._log("Checking system prerequisites...")
        
        # Check Conda
        if not self._check_command_available('conda'):
            self._log("ERROR: Conda is not installed or not in PATH")
            self._log("Please install Miniconda or Anaconda first:")
            self._log("https://docs.conda.io/en/latest/miniconda.html")
            return False
        self._log("Conda found - OK")
        
        # Check Git
        if not self._check_command_available('git'):
            self._log("ERROR: Git is not installed or not in PATH")
            self._log("Please install Git first:")
            self._log("https://git-scm.com/downloads")
            return False
        self._log("Git found - OK")
        
        # Check if installation directory is valid
        try:
            self.installation_path.parent.mkdir(parents=True, exist_ok=True)
            self._log(f"Installation directory prepared: {self.installation_path}")
        except Exception as e:
            self._log(f"ERROR: Cannot create installation directory: {str(e)}")
            return False
            
        return True
        
    def _clone_repository(self) -> bool:
        """Clone the LeRobot repository."""
        self._log("Cloning LeRobot repository from GitHub...")
        
        if self.installation_path.exists():
            self._log(f"Directory {self.installation_path} already exists, removing...")
            try:
                shutil.rmtree(self.installation_path)
            except Exception as e:
                self._log(f"ERROR: Could not remove existing directory: {str(e)}")
                return False
                
        command = f'git clone https://github.com/huggingface/lerobot.git "{self.installation_path}"'
        return self._run_command(command)
        
    def _create_environment(self) -> bool:
        """Create conda environment with Python 3.10."""
        self._log("Creating conda environment 'lerobot' with Python 3.10...")
        command = 'conda create -y -n lerobot python=3.10'
        return self._run_command(command)
        
    def _install_ffmpeg(self) -> bool:
        """Install FFmpeg in the conda environment."""
        self._log("Installing FFmpeg from conda-forge...")
        command = 'conda install -y ffmpeg -c conda-forge -n lerobot'
        return self._run_command(command)
        
    def _install_lerobot(self) -> bool:
        """Install LeRobot package in development mode."""
        self._log("Installing LeRobot package...")
        command = 'conda run -n lerobot pip install -e .'
        return self._run_command(command, cwd=self.installation_path)
        
    def _run_command(self, command: str, cwd: Optional[Path] = None) -> bool:
        """Execute a shell command and log output."""
        if not self.is_running:
            return False
            
        try:
            self._log(f"Executing: {command}")
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd,
                shell=True
            )
            
            # Read output line by line
            while True:
                if not self.is_running:
                    process.terminate()
                    return False
                    
                line = process.stdout.readline()
                if not line:
                    break
                    
                self._log(line.rstrip())
                
            process.wait()
            
            if process.returncode != 0:
                self._log(f"Command failed with exit code: {process.returncode}")
                return False
                
            return True
            
        except Exception as e:
            self._log(f"Error executing command: {str(e)}")
            return False
            
    def _check_command_available(self, command: str) -> bool:
        """Check if a command is available in the system PATH."""
        try:
            result = subprocess.run(
                [command, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
            
    def _log(self, message: str):
        """Send log message to UI callback."""
        if self.on_log_message:
            self.on_log_message(message)
            
    def _update_progress(self, step: InstallationStep, message: str):
        """Send progress update to UI callback."""
        if self.on_progress_update:
            self.on_progress_update(step, message)
            
    def _complete_installation(self, result: InstallationResult):
        """Send completion notification to UI callback."""
        if self.on_completion:
            self.on_completion(result)
            
    def _detect_usb_ports(self) -> bool:
        """Automatically detect USB ports for robotic arms."""
        self._log("Setting up automatic USB port detection...")
        
        if not PYSERIAL_AVAILABLE:
            self._log("Installing pyserial for USB port detection...")
            install_cmd = 'conda run -n lerobot pip install pyserial'
            if not self._run_command(install_cmd):
                self._log("Warning: Failed to install pyserial. USB port detection will be skipped.")
                self._log("You can manually install pyserial later with: conda activate lerobot && pip install pyserial")
                return True  # Don't fail installation for this
            
            # Try importing again after installation
            try:
                import serial
                import serial.tools.list_ports
                global PYSERIAL_AVAILABLE
                PYSERIAL_AVAILABLE = True
            except ImportError:
                self._log("Warning: pyserial installation failed. Port detection skipped.")
                return True
        
        try:
            # Get available ports
            available_ports = self._get_available_ports()
            
            if len(available_ports) == 0:
                self._log("No USB serial ports detected. Make sure your robotic arms are connected.")
                self._log("You can run port detection later using: python lerobot/find_port.py")
                return True
            elif len(available_ports) == 1:
                self._log("Only one USB port detected. You'll need to connect both arms for automatic detection.")
                self._log("You can run port detection later using: python lerobot/find_port.py")
                return True
            elif len(available_ports) >= 2:
                self._log(f"Found {len(available_ports)} USB serial ports:")
                for i, port in enumerate(available_ports, 1):
                    self._log(f"  {i}. {port['device']} - {port['description']}")
                
                # Create simple auto-configuration for common scenarios
                if len(available_ports) == 2:
                    self._log("Detected exactly 2 USB ports - setting up automatic configuration...")
                    self._create_auto_port_config(available_ports)
                else:
                    self._log(f"Multiple ports detected ({len(available_ports)}). Run interactive detection:")
                    self._log("  python lerobot/find_port.py")
                    
                # Copy the port detection tool and setup scripts to the installation directory
                self._copy_port_detection_tool()
                self._copy_setup_scripts()
                
            return True
            
        except Exception as e:
            self._log(f"USB port detection failed: {str(e)}")
            self._log("You can run port detection manually later using: python lerobot/find_port.py")
            return True  # Don't fail installation for port detection issues
    
    def _get_available_ports(self) -> List[Dict[str, str]]:
        """Get list of available serial ports with details."""
        if not PYSERIAL_AVAILABLE:
            return []
            
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
    
    def _create_auto_port_config(self, ports: List[Dict[str, str]]):
        """Create automatic port configuration for exactly 2 detected ports."""
        config_content = f'''# LeRobot Port Configuration
# Generated automatically after installation
# This is a default configuration - you may need to adjust ports based on your setup

# Automatically detected ports:
LEADER_ARM_PORT = "{ports[0]['device']}"
FOLLOWER_ARM_PORT = "{ports[1]['device']}"

# Port Details:
# Port 1: {ports[0]['device']} - {ports[0]['description']}
# Port 2: {ports[1]['device']} - {ports[1]['description']}

# IMPORTANT: Verify these port assignments match your actual setup!
# If the assignments are incorrect, run: python lerobot/find_port.py
'''
        
        config_file = self.installation_path / "lerobot_ports.py"
        try:
            with open(config_file, 'w') as f:
                f.write(config_content)
            self._log(f"✅ Auto-generated port configuration saved to: {config_file}")
            self._log("⚠️  IMPORTANT: Please verify these port assignments are correct!")
            self._log("   If incorrect, run: python lerobot/find_port.py")
        except Exception as e:
            self._log(f"Failed to save auto-configuration: {e}")
    
    def _copy_port_detection_tool(self):
        """Copy the port detection tool to the installation directory."""
        try:
            utils_dir = Path(__file__).parent.parent / "utils"
            source_file = utils_dir / "lerobot" / "find_port.py"
            target_dir = self.installation_path / "lerobot"
            
            if source_file.exists():
                target_dir.mkdir(exist_ok=True)
                shutil.copy2(source_file, target_dir / "find_port.py")
                self._log("✅ Port detection tool copied to installation directory")
                self._log("   Run interactive detection with: python lerobot/find_port.py")
            else:
                self._log("⚠️  Port detection tool not found in utils directory")
                
        except Exception as e:
            self._log(f"Failed to copy port detection tool: {e}")
            
    def _copy_setup_scripts(self):
        """Copy setup scripts to the installation directory."""
        try:
            # Just create a simple setup completion message since we moved away from extra scripts
            setup_content = '''#!/usr/bin/env python3
"""
LeRobot Installation Complete!

Your LeRobot environment is ready to use.
"""

print("🎉 LeRobot Installation Complete!")
print()
print("Next steps:")
print("1. conda activate lerobot")
print("2. python lerobot/find_port.py  # Configure USB ports")
print("3. Start building with LeRobot!")
'''
            
            setup_file = self.installation_path / "setup_complete.py"
            with open(setup_file, 'w') as f:
                f.write(setup_content)
            
            self._log("✅ Setup completion script created")
            self._log("   Run: python setup_complete.py for next steps")
                
        except Exception as e:
            self._log(f"Failed to create setup scripts: {e}")

    def get_installation_info(self) -> Dict[str, Any]:
        """Get information about the installation process."""
        return {
            'is_running': self.is_running,
            'current_step': self.current_step.value if self.current_step else None,
            'installation_path': str(self.installation_path) if self.installation_path else None
        }