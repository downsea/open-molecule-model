"""
Centralized logging system for PanGu Drug Model.

This module provides a unified logging interface for all components
of the PanGu Drug Model project with configurable log levels,
file rotation, and structured output.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

class PanGuLogger:
    """Centralized logger for PanGu Drug Model with file rotation and structured output."""
    
    def __init__(self, 
                 name: str = "PanGuDrugModel",
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 max_file_size: int = 50 * 1024 * 1024,  # 50MB
                 backup_count: int = 5):
        """
        Initialize the PanGu Drug Model logger.
        
        Args:
            name: Logger name
            log_dir: Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            max_file_size: Maximum size of log files before rotation (bytes)
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different log types
        (self.log_dir / "processing").mkdir(exist_ok=True)
        (self.log_dir / "training").mkdir(exist_ok=True)
        (self.log_dir / "evaluation").mkdir(exist_ok=True)
        (self.log_dir / "data_analysis").mkdir(exist_ok=True)
        (self.log_dir / "errors").mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup formatters
        self.file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        self.console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Setup handlers
        self.setup_handlers(max_file_size, backup_count)
        
    def setup_handlers(self, max_file_size: int, backup_count: int):
        """Setup file and console handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)
        
        # Main file handler with rotation
        main_log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        from logging.handlers import RotatingFileHandler
        
        file_handler = RotatingFileHandler(
            main_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(file_handler)
        
    def get_task_logger(self, task_name: str) -> logging.Logger:
        """Get a task-specific logger with dedicated file."""
        task_logger = logging.getLogger(f"{self.name}.{task_name}")
        
        # Remove existing handlers to avoid duplication
        task_logger.handlers.clear()
        
        # Task-specific file handler
        task_log_file = self.log_dir / task_name / f"{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        from logging.handlers import RotatingFileHandler
        
        task_file_handler = RotatingFileHandler(
            task_log_file,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=3,
            encoding='utf-8'
        )
        task_file_handler.setLevel(logging.DEBUG)
        task_file_handler.setFormatter(self.file_formatter)
        task_logger.addHandler(task_file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.console_formatter)
        task_logger.addHandler(console_handler)
        
        return task_logger
    
    def log_config(self, config: Dict[str, Any], task_name: str = "general"):
        """Log configuration parameters in a structured way."""
        self.logger.info(f"{task_name} configuration: {json.dumps(config, indent=2)}")
    
    def log_error_details(self, error: Exception, context: str = ""):
        """Log detailed error information."""
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
        
        # Also log to error-specific file
        error_log_file = self.log_dir / "errors" / f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(error_log_file, 'w') as f:
            f.write(f"Error Context: {context}\n")
            f.write(f"Error Type: {type(error).__name__}\n")
            f.write(f"Error Message: {str(error)}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    
    def log_processing_stats(self, stats: Dict[str, Any], task_name: str = "processing"):
        """Log processing statistics in a structured format."""
        self.logger.info(f"Processing statistics for {task_name}: {json.dumps(stats, indent=2)}")
        
        # Save to task-specific stats file
        task_dir = self.log_dir / task_name
        task_dir.mkdir(exist_ok=True)
        stats_file = task_dir / f"{task_name}_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'task': task_name,
                'statistics': stats
            }, f, indent=2)
    
    def log_progress(self, step: int, total: int, message: str = ""):
        """Log progress with percentage."""
        percentage = (step / max(total, 1)) * 100
        self.logger.info(f"Progress: {step}/{total} ({percentage:.1f}%) - {message}")
    
    def close(self):
        """Close all handlers."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

# Global logger instance
_global_logger = None

def get_logger(name: str = "PanGuDrugModel", 
               log_dir: str = "logs",
               log_level: str = "INFO") -> PanGuLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = PanGuLogger(name, log_dir, log_level)
    return _global_logger

def setup_task_logger(task_name: str, log_level: str = "INFO") -> logging.Logger:
    """Setup a task-specific logger."""
    logger = get_logger()
    return logger.get_task_logger(task_name)

# Convenience functions
def log_info(message: str, task_name: str = "general"):
    """Log info message."""
    logger = get_logger()
    logger.logger.info(f"[{task_name}] {message}")

def log_warning(message: str, task_name: str = "general"):
    """Log warning message."""
    logger = get_logger()
    logger.logger.warning(f"[{task_name}] {message}")

def log_error(message: str, task_name: str = "general"):
    """Log error message."""
    logger = get_logger()
    logger.logger.error(f"[{task_name}] {message}")

def log_debug(message: str, task_name: str = "general"):
    """Log debug message."""
    logger = get_logger()
    logger.logger.debug(f"[{task_name}] {message}")

if __name__ == "__main__":
    # Test the logging system
    logger = get_logger("TestLogger", "test_logs")
    logger.logger.info("Logger test successful")
    
    # Test task-specific logger
    train_logger = setup_task_logger("training")
    train_logger.info("Training logger test successful")
    
    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.log_error_details(e, "test_context")