"""
Centralized error handling and logging for the LeRobot installer.
Provides consistent error reporting and user-friendly error messages.
"""

import traceback
import sys
from typing import Optional, Callable
from enum import Enum


class ErrorType(Enum):
    """Types of errors that can occur during installation."""
    SYSTEM_ERROR = "system_error"
    NETWORK_ERROR = "network_error"
    PERMISSION_ERROR = "permission_error"
    DEPENDENCY_ERROR = "dependency_error"
    USER_ERROR = "user_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InstallationError(Exception):
    """Custom exception class for installation-specific errors."""
    
    def __init__(self, message: str, error_type: ErrorType = ErrorType.UNKNOWN_ERROR, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 suggestion: str = None, original_error: Exception = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.severity = severity
        self.suggestion = suggestion
        self.original_error = original_error


class ErrorHandler:
    """
    Centralized error handler that provides consistent error reporting
    and user-friendly error messages.
    """
    
    def __init__(self, log_callback: Optional[Callable[[str], None]] = None):
        self.log_callback = log_callback
        
    def handle_error(self, error: Exception, context: str = "") -> InstallationError:
        """
        Handle an error and convert it to a user-friendly InstallationError.
        
        Args:
            error: The original exception
            context: Additional context about when/where the error occurred
            
        Returns:
            InstallationError with user-friendly message and suggestions
        """
        # Log the original error for debugging
        self._log_error(error, context)
        
        # Convert to InstallationError with user-friendly message
        installation_error = self._convert_to_installation_error(error, context)
        
        return installation_error
        
    def _convert_to_installation_error(self, error: Exception, context: str) -> InstallationError:
        """Convert a generic exception to an InstallationError with helpful information."""
        error_str = str(error).lower()
        
        # Network-related errors
        if any(keyword in error_str for keyword in ['network', 'connection', 'timeout', 'unreachable', 'dns']):
            return InstallationError(
                message="Network connection error occurred",
                error_type=ErrorType.NETWORK_ERROR,
                severity=ErrorSeverity.HIGH,
                suggestion="Please check your internet connection and try again. If you're behind a corporate firewall, contact your IT administrator.",
                original_error=error
            )
            
        # Permission-related errors
        elif any(keyword in error_str for keyword in ['permission', 'access', 'denied', 'forbidden']):
            return InstallationError(
                message="Permission denied during installation",
                error_type=ErrorType.PERMISSION_ERROR,
                severity=ErrorSeverity.HIGH,
                suggestion="Try running the installer as administrator, or choose a different installation directory where you have write permissions.",
                original_error=error
            )
            
        # Git-related errors
        elif any(keyword in error_str for keyword in ['git', 'clone', 'repository']):
            return InstallationError(
                message="Git repository operation failed",
                error_type=ErrorType.DEPENDENCY_ERROR,
                severity=ErrorSeverity.HIGH,
                suggestion="Please ensure Git is installed and available in your system PATH. You can download Git from https://git-scm.com/downloads",
                original_error=error
            )
            
        # Conda-related errors
        elif any(keyword in error_str for keyword in ['conda', 'environment', 'package']):
            return InstallationError(
                message="Conda environment operation failed",
                error_type=ErrorType.DEPENDENCY_ERROR,
                severity=ErrorSeverity.HIGH,
                suggestion="Please ensure Conda or Miniconda is installed and available in your system PATH. You can download Miniconda from https://docs.conda.io/en/latest/miniconda.html",
                original_error=error
            )
            
        # Disk space errors
        elif any(keyword in error_str for keyword in ['space', 'disk', 'storage', 'full']):
            return InstallationError(
                message="Insufficient disk space for installation",
                error_type=ErrorType.SYSTEM_ERROR,
                severity=ErrorSeverity.HIGH,
                suggestion="Please free up disk space or choose a different installation directory with more available space.",
                original_error=error
            )
            
        # Python/pip errors
        elif any(keyword in error_str for keyword in ['pip', 'python', 'module', 'import']):
            return InstallationError(
                message="Python package installation failed",
                error_type=ErrorType.DEPENDENCY_ERROR,
                severity=ErrorSeverity.MEDIUM,
                suggestion="There may be a conflict with existing Python packages. Try creating a fresh conda environment or contact support.",
                original_error=error
            )
            
        # File system errors
        elif any(keyword in error_str for keyword in ['file', 'directory', 'path', 'exists']):
            return InstallationError(
                message="File system operation failed",
                error_type=ErrorType.SYSTEM_ERROR,
                severity=ErrorSeverity.MEDIUM,
                suggestion="Please check that the installation path is valid and you have the necessary permissions.",
                original_error=error
            )
            
        # Default case for unknown errors
        else:
            return InstallationError(
                message=f"An unexpected error occurred: {str(error)}",
                error_type=ErrorType.UNKNOWN_ERROR,
                severity=ErrorSeverity.MEDIUM,
                suggestion="Please try the installation again. If the problem persists, check the installation log for more details.",
                original_error=error
            )
            
    def _log_error(self, error: Exception, context: str):
        """Log error details for debugging purposes."""
        if not self.log_callback:
            return
            
        error_message = f"ERROR in {context}: {str(error)}"
        self.log_callback(error_message)
        
        # Log traceback for debugging
        if hasattr(error, '__traceback__') and error.__traceback__:
            traceback_lines = traceback.format_exception(
                type(error), error, error.__traceback__
            )
            for line in traceback_lines:
                self.log_callback(line.rstrip())
                
    def get_user_friendly_message(self, error: InstallationError) -> str:
        """
        Generate a user-friendly error message with suggestions.
        
        Args:
            error: The InstallationError to format
            
        Returns:
            Formatted error message for display to user
        """
        message_parts = [error.message]
        
        if error.suggestion:
            message_parts.append(f"\nSuggestion: {error.suggestion}")
            
        if error.original_error and str(error.original_error) != error.message:
            message_parts.append(f"\nTechnical details: {str(error.original_error)}")
            
        return "\n".join(message_parts)
        
    def should_retry(self, error: InstallationError) -> bool:
        """
        Determine if an operation should be retried based on the error type.
        
        Args:
            error: The InstallationError to evaluate
            
        Returns:
            True if the operation can be safely retried
        """
        # Network errors can often be retried
        if error.error_type == ErrorType.NETWORK_ERROR:
            return True
            
        # Some system errors might be temporary
        if error.error_type == ErrorType.SYSTEM_ERROR and error.severity != ErrorSeverity.CRITICAL:
            return True
            
        # Don't retry permission errors or user errors
        if error.error_type in [ErrorType.PERMISSION_ERROR, ErrorType.USER_ERROR]:
            return False
            
        # Don't retry dependency errors (need to be fixed first)
        if error.error_type == ErrorType.DEPENDENCY_ERROR:
            return False
            
        # Default to no retry for unknown errors
        return False