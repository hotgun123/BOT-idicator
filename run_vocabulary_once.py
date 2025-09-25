#!/usr/bin/env python3
"""
Script chạy vocabulary bot một lần cho GitHub Actions
"""

import logging
from vocabulary_bot import run_vocabulary_bot_once

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🚀 Starting vocabulary bot for GitHub Actions...")
    
    try:
        success = run_vocabulary_bot_once()
        
        if success:
            logger.info("✅ Vocabulary bot completed successfully!")
            exit(0)
        else:
            logger.error("❌ Vocabulary bot failed!")
            exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error running vocabulary bot: {e}")
        exit(1)
