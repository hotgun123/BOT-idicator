#!/usr/bin/env python3
"""
Test chỉ gửi Telegram - không có phân tích
"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7496162935:AAGncIsO4q18cOWRGpK0vYb_5zWxYNEgWKQ")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1866335373")

def test_telegram():
    """Test gửi Telegram"""
    print(f"Bot Token: {TELEGRAM_BOT_TOKEN[:20]}...")
    print(f"Chat ID: {TELEGRAM_CHAT_ID}")
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    message = f"""
🤖 <b>TELEGRAM TEST ONLY</b>

✅ <b>Test từ GitHub Actions</b>
⏰ Thời gian: {current_time}
🌐 Environment: GitHub Actions

🔧 <b>Debug Info:</b>
• Bot Token: {TELEGRAM_BOT_TOKEN[:20]}...
• Chat ID: {TELEGRAM_CHAT_ID}
• Test: Simple message

🎉 <b>Nếu bạn thấy message này, Telegram hoạt động!</b>
    """
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    
    print(f"Sending to URL: {url}")
    print(f"Data: {data}")
    
    try:
        response = requests.post(url, data=data, timeout=30)
        print(f"Response Status: {response.status_code}")
        print(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            print("✅ Gửi message thành công!")
            return True
        else:
            print(f"❌ Lỗi: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Bắt đầu test Telegram...")
    success = test_telegram()
    if success:
        print("🎉 Test thành công!")
    else:
        print("❌ Test thất bại!")
