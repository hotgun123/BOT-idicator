#!/usr/bin/env python3
"""
Script test nhanh - chỉ phân tích cơ bản và gửi Telegram
Chạy trong 5 phút
"""

import ccxt
import numpy as np
import pandas as pd
import requests
import json
import os
from datetime import datetime
import time
import logging
from dotenv import load_dotenv
import ta

# Load environment variables
load_dotenv()

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7496162935:AAGncIsO4q18cOWRGpK0vYb_5zWxYNEgWKQ")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1866335373")

# Khởi tạo kết nối với Binance
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
    }
})

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

def get_quick_analysis(symbol):
    """Phân tích nhanh cho một symbol"""
    try:
        logger.info(f"🔍 Đang phân tích {symbol}...")
        
        # Lấy dữ liệu 1h
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=50)
        if not ohlcv:
            return None
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Tính toán indicators cơ bản
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        
        # Lấy giá trị cuối cùng
        current_price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        ema_20 = df['ema_20'].iloc[-1]
        ema_50 = df['ema_50'].iloc[-1]
        
        # Tín hiệu đơn giản
        signal = "NEUTRAL"
        if rsi < 30 and current_price > ema_20:
            signal = "LONG"
        elif rsi > 70 and current_price < ema_20:
            signal = "SHORT"
        
        return {
            'symbol': symbol,
            'price': current_price,
            'rsi': rsi,
            'signal': signal,
            'ema_20': ema_20,
            'ema_50': ema_50
        }
        
    except Exception as e:
        logger.error(f"❌ Lỗi phân tích {symbol}: {e}")
        return None

def main():
    """Main function nhanh"""
    logger.info("🚀 Bắt đầu phân tích nhanh...")
    
    # Phân tích BTC và ETH
    symbols = ['BTC/USDT', 'ETH/USDT']
    results = []
    
    for symbol in symbols:
        result = get_quick_analysis(symbol)
        if result:
            results.append(result)
            logger.info(f"✅ Đã phân tích {symbol}")
    
    if results:
        # Tạo message
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"""
🤖 <b>PHÂN TÍCH NHANH</b>

⏰ Thời gian: {current_time}
🌐 Environment: GitHub Actions

📊 <b>KẾT QUẢ:</b>

"""
        
        for result in results:
            message += f"""
<b>{result['symbol']}</b>
💰 Giá: ${result['price']:,.2f}
📈 RSI: {result['rsi']:.2f}
🎯 Tín hiệu: <b>{result['signal']}</b>
📉 EMA20: ${result['ema_20']:,.2f}
📉 EMA50: ${result['ema_50']:,.2f}

"""
        
        message += """
✅ <b>Phân tích hoàn thành!</b>
⚡ Chạy trong 2 phút
📱 Bot hoạt động bình thường
"""
        
        # Gửi về Telegram
        success = send_telegram_message(message)
        
        if success:
            logger.info("🎉 Gửi Telegram thành công!")
        else:
            logger.error("❌ Lỗi gửi Telegram")
    else:
        logger.error("❌ Không có kết quả")
    
    logger.info("🏁 Hoàn thành!")

if __name__ == "__main__":
    main()
