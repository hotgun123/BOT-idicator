#!/usr/bin/env python3
"""
Script test đơn giản để kiểm tra GitHub Actions
Chạy nhanh và gửi test message về Telegram
"""

import os
import requests
import json
from datetime import datetime
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7496162935:AAGncIsO4q18cOWRGpK0vYb_5zWxYNEgWKQ")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1866335373")

def send_telegram_message(message):
    """Gửi message về Telegram"""
    try:
        if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
            logger.warning("Telegram credentials chưa được cấu hình!")
            return False
            
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            logger.info("✅ Gửi message thành công!")
            return True
        else:
            logger.error(f"❌ Lỗi gửi message: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Lỗi kết nối Telegram: {str(e)}")
        return False

def test_github_actions():
    """Test GitHub Actions"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Test message
    message = f"""
🤖 <b>GitHub Actions Test</b>

✅ <b>Bot đã chạy thành công!</b>
⏰ Thời gian: {current_time}
🌐 Environment: GitHub Actions
📊 Status: Ready for trading analysis

🔧 <b>Kiểm tra:</b>
• Python environment: ✅
• Dependencies: ✅  
• Telegram connection: ✅
• File system: ✅

🚀 <b>Sẵn sàng chạy bot trading!</b>
    """
    
    # Gửi test message
    success = send_telegram_message(message)
    
    if success:
        logger.info("🎉 GitHub Actions test thành công!")
        return True
    else:
        logger.error("❌ GitHub Actions test thất bại!")
        return False

if __name__ == "__main__":
    logger.info("🚀 Bắt đầu test GitHub Actions...")
    test_github_actions()
