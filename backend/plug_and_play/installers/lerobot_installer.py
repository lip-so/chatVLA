#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import subprocess
import threading
import os
import sys
import shutil
from pathlib import Path

class LeRobotInstaller:
    def __init__(self, root):
        self.root = root
        self.root.title("LeRobot Installation Assistant")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Fix for macOS rendering issues
        if sys.platform == "darwin":
            self.root.tk.call('tk', 'scaling', 1.0)
            self.root.configure(bg='#f0f0f0')
        
        self.install_path = None
        self.is_installing = False
        
        # Force update before setting up UI
        self.root.update_idletasks()
        self.setup_ui()
        
    def setup_ui(self):
        # Configure root grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure main_frame grid
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Force frame to show
        main_frame.update_idletasks()
        
        # Title with macOS-safe font
        title_font = ("Helvetica", 16, "bold") if sys.platform == "darwin" else ("Arial", 16, "bold")
        ttk.Label(main_frame, text="LeRobot Installation Assistant", 
                 font=title_font).grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        ttk.Label(main_frame, text="Installation Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        path_frame = ttk.Frame(main_frame)
        path_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        path_frame.columnconfigure(0, weight=1)
        
        self.path_var = tk.StringVar(value=str(Path.home() / "lerobot"))
        self.path_entry = ttk.Entry(path_frame, textvariable=self.path_var)
        self.path_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(path_frame, text="Browse", 
                  command=self.browse_directory).grid(row=0, column=1)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        self.install_button = ttk.Button(button_frame, text="Start Installation", 
                                        command=self.start_installation, style="Accent.TButton")
        self.install_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.cancel_button = ttk.Button(button_frame, text="Cancel", 
                                       command=self.cancel_installation, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT)
        
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(main_frame, height=20, width=80)
        self.log_text.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_label = ttk.Label(status_frame, text="Ready to install LeRobot")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
    def browse_directory(self):
        directory = filedialog.askdirectory(initialdir=self.path_var.get())
        if directory:
            self.path_var.set(directory)
            
    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def update_status(self, status):
        self.status_label.config(text=status)
        self.root.update_idletasks()
        
    def start_installation(self):
        if self.is_installing:
            return
            
        self.install_path = Path(self.path_var.get())
        
        if not messagebox.askyesno("Confirm Installation", 
                                  f"This will install LeRobot to:\n{self.install_path}\n\nContinue?"):
            return
            
        self.is_installing = True
        self.install_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.progress.start()
        
        self.log_text.delete(1.0, tk.END)
        self.log_message("Starting LeRobot installation...")
        
        installation_thread = threading.Thread(target=self.run_installation)
        installation_thread.daemon = True
        installation_thread.start()
        
    def cancel_installation(self):
        if messagebox.askyesno("Cancel Installation", "Are you sure you want to cancel the installation?"):
            self.is_installing = False
            self.finish_installation("Installation cancelled by user")
            
    def finish_installation(self, message):
        self.is_installing = False
        self.progress.stop()
        self.install_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        self.update_status(message)
        
    def run_command(self, command, cwd=None, shell=True):
        if not self.is_installing:
            return False
            
        try:
            self.log_message(f"Running: {command}")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd,
                shell=shell
            )
            
            for line in iter(process.stdout.readline, ''):
                if not self.is_installing:
                    process.terminate()
                    return False
                self.log_message(line.rstrip())
                
            process.wait()
            
            if process.returncode != 0:
                self.log_message(f"Command failed with return code: {process.returncode}")
                return False
                
            return True
            
        except Exception as e:
            self.log_message(f"Error running command: {str(e)}")
            return False
            
    def check_conda(self):
        try:
            result = subprocess.run(['conda', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
            
    def run_installation(self):
        try:
            self.update_status("Checking prerequisites...")
            
            if not self.check_conda():
                self.log_message("ERROR: Conda is not installed or not in PATH")
                self.log_message("Please install Miniconda or Anaconda first:")
                self.log_message("https://docs.conda.io/en/latest/miniconda.html")
                self.finish_installation("Installation failed: Conda not found")
                return
                
            self.log_message("Conda found ✓")
            
            if not shutil.which('git'):
                self.log_message("ERROR: Git is not installed or not in PATH")
                self.log_message("Please install Git first: https://git-scm.com/downloads")
                self.finish_installation("Installation failed: Git not found")
                return
                
            self.log_message("Git found ✓")
            
            self.install_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.update_status("Cloning LeRobot repository...")
            if not self.run_command(
                f'git clone https://github.com/huggingface/lerobot.git "{self.install_path}"'
            ):
                self.finish_installation("Installation failed during git clone")
                return
                
            self.update_status("Creating conda environment...")
            if not self.run_command('conda create -y -n lerobot python=3.10'):
                self.finish_installation("Installation failed during conda environment creation")
                return
                
            self.update_status("Installing ffmpeg...")
            if not self.run_command('conda install -y ffmpeg -c conda-forge -n lerobot'):
                self.finish_installation("Installation failed during ffmpeg installation")
                return
                
            self.update_status("Installing LeRobot...")
            conda_prefix = self.get_conda_prefix()
            if conda_prefix:
                pip_path = conda_prefix / "envs" / "lerobot" / "bin" / "pip"
                if not pip_path.exists() and os.name == 'nt':
                    pip_path = conda_prefix / "envs" / "lerobot" / "Scripts" / "pip.exe"
                    
                if pip_path.exists():
                    if not self.run_command(f'"{pip_path}" install -e .', cwd=self.install_path):
                        self.finish_installation("Installation failed during pip install")
                        return
                else:
                    if not self.run_command('conda run -n lerobot pip install -e .', cwd=self.install_path):
                        self.finish_installation("Installation failed during pip install")
                        return
            else:
                if not self.run_command('conda run -n lerobot pip install -e .', cwd=self.install_path):
                    self.finish_installation("Installation failed during pip install")
                    return

            # USB Port Detection
            self.update_status("Setting up USB port detection...")
            self.setup_usb_port_detection()
                    
            self.log_message("\n" + "="*50)
            self.log_message("Installation completed successfully!")
            self.log_message("="*50)
            self.log_message(f"LeRobot has been installed to: {self.install_path}")
            self.log_message("\nTo use LeRobot:")
            self.log_message("1. Open a terminal/command prompt")
            self.log_message("2. Run: conda activate lerobot")
            self.log_message("3. Navigate to your project directory")
            self.log_message("4. Start using LeRobot!")
            
            self.finish_installation("Installation completed successfully!")
            
        except Exception as e:
            self.log_message(f"Unexpected error: {str(e)}")
            self.finish_installation("Installation failed with unexpected error")
            
    def setup_usb_port_detection(self):
        """Set up USB port detection functionality."""
        try:
            self.log_message("Setting up automatic USB port detection...")
            
            # Install pyserial if not available
            self.log_message("Installing pyserial for USB port detection...")
            if not self.run_command('conda run -n lerobot pip install pyserial'):
                self.log_message("Warning: Failed to install pyserial. USB port detection will be skipped.")
                return
            
            # Copy port detection tool to installation directory
            source_dir = Path(__file__).parent / "lerobot"
            target_dir = self.install_path / "lerobot"
            
            if source_dir.exists() and (source_dir / "find_port.py").exists():
                target_dir.mkdir(exist_ok=True)
                shutil.copy2(source_dir / "find_port.py", target_dir / "find_port.py")
                self.log_message("✅ Port detection tool copied to installation directory")
                self.log_message("   Run interactive detection with: python lerobot/find_port.py")
            else:
                self.log_message("⚠️  Port detection tool not found in source directory")
            
            # Try to detect available ports
            try:
                import serial.tools.list_ports
                available_ports = list(serial.tools.list_ports.comports())
                
                if len(available_ports) == 0:
                    self.log_message("No USB serial ports detected. Connect your robotic arms and run:")
                    self.log_message("  python lerobot/find_port.py")
                elif len(available_ports) == 1:
                    self.log_message("Only one USB port detected. Connect both arms and run:")
                    self.log_message("  python lerobot/find_port.py")
                elif len(available_ports) == 2:
                    self.log_message(f"Found {len(available_ports)} USB serial ports - creating auto-configuration...")
                    self.create_auto_port_config(available_ports)
                else:
                    self.log_message(f"Found {len(available_ports)} USB ports. Run interactive detection:")
                    self.log_message("  python lerobot/find_port.py")
                    
            except ImportError:
                self.log_message("Port detection requires pyserial. Run interactive detection later:")
                self.log_message("  python lerobot/find_port.py")
            
        except Exception as e:
            self.log_message(f"USB port detection setup failed: {str(e)}")
            self.log_message("You can run port detection manually later using: python lerobot/find_port.py")
    
    def create_auto_port_config(self, ports):
        """Create automatic port configuration for exactly 2 detected ports."""
        config_content = f'''# LeRobot Port Configuration
# Generated automatically after installation
# This is a default configuration - you may need to adjust ports based on your setup

# Automatically detected ports:
LEADER_ARM_PORT = "{ports[0].device}"
FOLLOWER_ARM_PORT = "{ports[1].device}"

# Port Details:
# Port 1: {ports[0].device} - {ports[0].description}
# Port 2: {ports[1].device} - {ports[1].description}

# IMPORTANT: Verify these port assignments match your actual setup!
# If the assignments are incorrect, run: python lerobot/find_port.py
'''
        
        config_file = self.install_path / "lerobot_ports.py"
        try:
            with open(config_file, 'w') as f:
                f.write(config_content)
            self.log_message(f"✅ Auto-generated port configuration saved to: lerobot_ports.py")
            self.log_message("⚠️  IMPORTANT: Please verify these port assignments are correct!")
            self.log_message("   If incorrect, run: python lerobot/find_port.py")
        except Exception as e:
            self.log_message(f"Failed to save auto-configuration: {e}")

    def get_conda_prefix(self):
        try:
            result = subprocess.run(['conda', 'info', '--base'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except:
            pass
        return None

def main():
    root = tk.Tk()
    
    # macOS-specific fixes
    if sys.platform == "darwin":
        # Try to use native macOS appearance
        try:
            root.tk.call('tk', 'scaling', 1.0)
            root.configure(bg='#f0f0f0')
        except:
            pass
    
    # Configure ttk style
    style = ttk.Style()
    
    # Use default theme on macOS
    if sys.platform == "darwin":
        try:
            style.theme_use('aqua')
        except:
            try:
                style.theme_use('default')
            except:
                pass
    
    # Configure accent button style
    try:
        style.configure('Accent.TButton', foreground='white', background='#0078d4')
    except:
        pass
    
    # Force root to show before creating app
    root.update_idletasks()
    
    app = LeRobotInstaller(root)
    
    def on_closing():
        if app.is_installing:
            if messagebox.askyesno("Exit", "Installation is in progress. Are you sure you want to exit?"):
                app.is_installing = False
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Final update before starting mainloop
    root.update()
    root.mainloop()

if __name__ == "__main__":
    main()