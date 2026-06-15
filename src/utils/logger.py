# rag-chatbot/src/utils/logger.py

import logging
import os
from configs import config

def setup_logger():
    """Set up the application logger."""
    if not os.path.exists(config.LOGS_DIR):
        os.makedirs(config.LOGS_DIR)

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE_PATH),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()