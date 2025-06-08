import logging
import sys
from pathlib import Path
from typing import Optional, Union
import colorlog  # pip install colorlog
from datetime import datetime

def setup_logger(
    name: str = __name__,
    log_file: Optional[Union[str, Path]] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    mode: str = 'a',
    propagate: bool = False
) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = propagate
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    console_format = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)-8s - %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    file_format = logging.Formatter(
        '%(asctime)s - %(levelname)-8s - %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

def configure_main_logger(
    log_dir: Union[str, Path] = "logs",
    log_prefix: str = "train",
    console_verbose: bool = True
) -> logging.Logger:

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{log_prefix}_{timestamp}.log"
    
    console_level = logging.DEBUG if console_verbose else logging.INFO
    
    return setup_logger(
        name="main",
        log_file=log_file,
        console_level=console_level,
        file_level=logging.DEBUG,
        mode='w'
    )

# if __name__ == "__main__":
#     # Инициализация
#     logger = setup_logger(log_file="test.log")
    
#     # Примеры сообщений
#     logger.debug("Это debug сообщение")
#     logger.info("Это info сообщение")
#     logger.warning("Это warning сообщение")
#     logger.error("Это error сообщение")
#     logger.critical("Это critical сообщение")
    
#     # Или с помощью configure_main_logger
#     main_logger = configure_main_logger()
#     main_logger.info("Логгер для основного приложения настроен!")