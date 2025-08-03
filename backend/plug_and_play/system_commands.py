#!/usr/bin/env python3
"""
Safe system command execution for LeRobot Plug & Play
Handles terminal commands with proper security and error handling.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

class SystemCommandError(Exception):
    """Custom exception for system command errors"""
    pass

class SafeSystemCommands:
    """
    Safe execution of system commands with security controls
    """
    
    # Allowed commands with their allowed arguments patterns
    ALLOWED_COMMANDS = {
        'git': [
            r'^clone\s+https://github\.com/[\w\-\.]+/[\w\-\.]+\.git\s+.+$',
            r'^status$',
            r'^log\s+--oneline\s+-n\s+\d+$'
        ],
        'conda': [
            r'^create\s+-n\s+\w+\s+python=3\.\d+\s+-y$',
            r'^env\s+remove\s+-n\s+\w+\s+-y$',
            r'^run\s+-n\s+\w+\s+pip\s+install\s+.+$',
            r'^list\s+-n\s+\w+$'
        ],
        'pip': [
            r'^install\s+[\w\-\.\[\]>=<,\s]+$',
            r'^list$',
            r'^show\s+\w+$'
        ],
        'python': [
            r'^-c\s+["\']import\s+\w+[;,\s\w]*["\']$',
            r'^-m\s+lerobot\.[\w\.]+\s+.+$'
        ],
        'ls': [r'^.*$'],  # Generally safe
        'pwd': [r'^$'],   # No arguments
        'which': [r'^\w+$']  # Single command name
    }
    
    # Blocked patterns (additional security)
    BLOCKED_PATTERNS = [
        r'sudo\s+',
        r'rm\s+-rf\s+/',
        r'chmod\s+777',
        r'&\s*$',  # Background execution
        r';\s*',   # Command chaining
        r'\|\s*',  # Piping
        r'`.*`',   # Command substitution
        r'\$\(',   # Command substitution
        r'>\s*/dev/',  # Device redirection
    ]
    
    def __init__(self, allowed_base_paths: List[str] = None):
        """
        Initialize with allowed base paths for file operations
        
        Args:
            allowed_base_paths: List of directory paths where operations are allowed
        """
        self.allowed_base_paths = allowed_base_paths or [
            str(Path.home()),
            '/tmp',
            '/opt/lerobot'
        ]
        
    def validate_command(self, command: str, args: List[str]) -> bool:
        """
        Validate if a command is safe to execute
        
        Args:
            command: The base command (e.g., 'git', 'conda')
            args: List of arguments
            
        Returns:
            bool: True if command is safe, False otherwise
        """
        # Check if command is in allowed list
        if command not in self.ALLOWED_COMMANDS:
            return False
            
        # Reconstruct full command for pattern matching
        full_command = ' '.join(args)
        
        # Check against blocked patterns first
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, full_command, re.IGNORECASE):
                return False
                
        # Check against allowed patterns
        allowed_patterns = self.ALLOWED_COMMANDS[command]
        for pattern in allowed_patterns:
            if re.match(pattern, full_command, re.IGNORECASE):
                return True
                
        return False
    
    def validate_path(self, path: str) -> bool:
        """
        Validate if a path is within allowed directories
        
        Args:
            path: File system path to validate
            
        Returns:
            bool: True if path is allowed, False otherwise
        """
        try:
            abs_path = Path(path).resolve()
            
            for allowed_base in self.allowed_base_paths:
                allowed_abs = Path(allowed_base).resolve()
                if abs_path.is_relative_to(allowed_abs):
                    return True
                    
            return False
        except:
            return False
    
    def run_command(
        self, 
        command: str, 
        args: List[str], 
        working_dir: Optional[str] = None,
        timeout: int = 300
    ) -> Tuple[int, str, str]:
        """
        Safely execute a system command
        
        Args:
            command: Base command to execute
            args: Command arguments
            working_dir: Working directory for execution
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (return_code, stdout, stderr)
            
        Raises:
            SystemCommandError: If command is not allowed or execution fails
        """
        # Validate command
        if not self.validate_command(command, args):
            raise SystemCommandError(f"Command not allowed: {command} {' '.join(args)}")
            
        # Validate working directory if provided
        if working_dir and not self.validate_path(working_dir):
            raise SystemCommandError(f"Working directory not allowed: {working_dir}")
            
        # Check if command exists
        if not shutil.which(command):
            raise SystemCommandError(f"Command not found: {command}")
            
        try:
            # Prepare command
            full_command = [command] + args
            
            # Execute command
            result = subprocess.run(
                full_command,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy()
            )
            
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            raise SystemCommandError(f"Command timed out after {timeout} seconds")
        except Exception as e:
            raise SystemCommandError(f"Command execution failed: {str(e)}")
    
    def stream_command(
        self,
        command: str,
        args: List[str],
        working_dir: Optional[str] = None,
        callback=None
    ) -> int:
        """
        Execute command with streaming output
        
        Args:
            command: Base command to execute
            args: Command arguments  
            working_dir: Working directory for execution
            callback: Function to call with each line of output
            
        Returns:
            int: Return code
            
        Raises:
            SystemCommandError: If command is not allowed or execution fails
        """
        # Validate command
        if not self.validate_command(command, args):
            raise SystemCommandError(f"Command not allowed: {command} {' '.join(args)}")
            
        # Validate working directory if provided
        if working_dir and not self.validate_path(working_dir):
            raise SystemCommandError(f"Working directory not allowed: {working_dir}")
            
        # Check if command exists
        if not shutil.which(command):
            raise SystemCommandError(f"Command not found: {command}")
            
        try:
            # Prepare command
            full_command = [command] + args
            
            # Execute with streaming
            process = subprocess.Popen(
                full_command,
                cwd=working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=os.environ.copy()
            )
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                if line and callback:
                    callback(line.strip())
            
            # Wait for completion
            process.wait()
            return process.returncode
            
        except Exception as e:
            raise SystemCommandError(f"Streaming command execution failed: {str(e)}")

# Global instance for the application
safe_commands = SafeSystemCommands()

def run_safe_command(command: str, args: List[str], **kwargs) -> Tuple[int, str, str]:
    """Convenience function for running safe commands"""
    return safe_commands.run_command(command, args, **kwargs)

def stream_safe_command(command: str, args: List[str], callback=None, **kwargs) -> int:
    """Convenience function for streaming safe commands"""
    return safe_commands.stream_command(command, args, callback=callback, **kwargs)