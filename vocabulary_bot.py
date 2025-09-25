#!/usr/bin/env python3
"""
Bot học từ vựng tiếng Anh độc lập
Gửi từ vựng B1-C1 qua Telegram riêng biệt
"""

import time
import schedule
import logging
from datetime import datetime
from vocabulary_learning import send_vocabulary_lesson
from vocabulary_config import VOCABULARY_SCHEDULE, BOT_INFO

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vocabulary_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def send_vocabulary_scheduled():
    """Gửi từ vựng theo lịch"""
    try:
        logger.info("📚 Bắt đầu gửi bài học từ vựng...")
        success = send_vocabulary_lesson()
        
        if success:
            logger.info("✅ Đã gửi bài học từ vựng thành công!")
        else:
            logger.error("❌ Không thể gửi bài học từ vựng")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi gửi từ vựng: {e}")
        return False

def setup_schedule():
    """Thiết lập lịch gửi từ vựng"""
    if not VOCABULARY_SCHEDULE['enabled']:
        logger.warning("⚠️ Lịch gửi từ vựng đã bị tắt")
        return
    
    # Lấy danh sách giờ gửi
    times = VOCABULARY_SCHEDULE['times']
    
    for hour in times:
        # Chuyển đổi giờ UTC sang định dạng schedule
        time_str = f"{hour:02d}:00"
        schedule.every().day.at(time_str).do(send_vocabulary_scheduled)
        logger.info(f"⏰ Đã lập lịch gửi từ vựng lúc {time_str} UTC")
    
    logger.info(f"📅 Đã lập lịch {len(times)} lần gửi từ vựng mỗi ngày")

def show_bot_info():
    """Hiển thị thông tin bot"""
    logger.info("=" * 50)
    logger.info(f"🤖 {BOT_INFO['name']}")
    logger.info(f"📝 {BOT_INFO['description']}")
    logger.info(f"🔢 Version: {BOT_INFO['version']}")
    logger.info(f"👨‍💻 Author: {BOT_INFO['author']}")
    logger.info("=" * 50)

def test_vocabulary_bot():
    """Test bot từ vựng"""
    logger.info("🧪 Testing vocabulary bot...")
    
    # Test gửi từ vựng
    success = send_vocabulary_scheduled()
    
    if success:
        logger.info("✅ Test thành công!")
        return True
    else:
        logger.error("❌ Test thất bại!")
        return False

def run_vocabulary_bot():
    """Chạy bot từ vựng"""
    logger.info("🚀 Khởi động Vocabulary Bot...")
    
    # Hiển thị thông tin bot
    show_bot_info()
    
    # Test bot trước
    logger.info("🧪 Testing bot trước khi chạy...")
    if not test_vocabulary_bot():
        logger.error("❌ Test thất bại, dừng bot")
        return
    
    # Thiết lập lịch
    setup_schedule()
    
    logger.info("🔄 Bot đang chạy... Nhấn Ctrl+C để dừng")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Kiểm tra mỗi phút
            
    except KeyboardInterrupt:
        logger.info("🛑 Bot đã dừng")
    except Exception as e:
        logger.error(f"❌ Lỗi bot: {e}")

def run_vocabulary_bot_once():
    """Chạy bot từ vựng một lần (cho GitHub Actions)"""
    logger.info("🚀 Chạy Vocabulary Bot một lần...")
    
    # Hiển thị thông tin bot
    show_bot_info()
    
    # Gửi từ vựng ngay lập tức
    success = send_vocabulary_scheduled()
    
    if success:
        logger.info("✅ Đã gửi từ vựng thành công!")
    else:
        logger.error("❌ Không thể gửi từ vựng")
    
    return success

if __name__ == "__main__":
    run_vocabulary_bot()
