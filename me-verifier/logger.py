import os
import sys
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_dir: str = None, log_filename: str = None) -> logging.Logger:
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    
    if log_filename is None:
        log_filename = f"{name}.log"
    
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    fh = RotatingFileHandler(logfile, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger