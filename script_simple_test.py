#!/usr/bin/env python3
"""
Script test đơn giản - chỉ phân tích cơ bản và gửi Telegram
Không có ML training để tránh timeout
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

def get_technical_indicators(data):
    """Tính toán technical indicators cơ bản"""
    try:
        df = pd.DataFrame(data)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        
        # EMA
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        
        return df
    except Exception as e:
        logger.error(f"❌ Lỗi tính toán indicators: {e}")
        return None

def analyze_symbol_simple(symbol):
    """Phân tích symbol đơn giản"""
    try:
        logger.info(f"🔍 Đang phân tích {symbol}...")
        
        # Lấy dữ liệu
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=100)
        if not ohlcv:
            logger.warning(f"⚠️ Không lấy được dữ liệu cho {symbol}")
            return None
        
        # Tính toán indicators
        df = get_technical_indicators(ohlcv)
        if df is None:
            return None
        
        # Lấy giá trị cuối cùng
        current_price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        ema_20 = df['ema_20'].iloc[-1]
        ema_50 = df['ema_50'].iloc[-1]
        
        # Phân tích tín hiệu đơn giản
        signal = "NEUTRAL"
        if rsi < 30 and macd > macd_signal and current_price > ema_20:
            signal = "LONG"
        elif rsi > 70 and macd < macd_signal and current_price < ema_20:
            signal = "SHORT"
        
        return {
            'symbol': symbol,
            'price': current_price,
            'rsi': rsi,
            'macd': macd,
            'signal': signal,
            'ema_20': ema_20,
            'ema_50': ema_50
        }
        
    except Exception as e:
        logger.error(f"❌ Lỗi phân tích {symbol}: {e}")
        return None

def main():
    """Main function đơn giản"""
    logger.info("🚀 Bắt đầu phân tích đơn giản...")
    
    # Phân tích BTC và ETH
    symbols = ['BTC/USDT', 'ETH/USDT']
    results = []
    
    for symbol in symbols:
        result = analyze_symbol_simple(symbol)
        if result:
            results.append(result)
            logger.info(f"✅ Đã phân tích {symbol}")
    
    if results:
        # Tạo message
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"""
🤖 <b>PHÂN TÍCH TRADING ĐƠN GIẢN</b>

⏰ Thời gian: {current_time}
🌐 Environment: GitHub Actions

📊 <b>KẾT QUẢ PHÂN TÍCH:</b>

"""
        
        for result in results:
            message += f"""
<b>{result['symbol']}</b>
💰 Giá hiện tại: ${result['price']:,.2f}
📈 RSI: {result['rsi']:.2f}
📊 MACD: {result['macd']:.4f}
🎯 Tín hiệu: <b>{result['signal']}</b>
📉 EMA20: ${result['ema_20']:,.2f}
📉 EMA50: ${result['ema_50']:,.2f}

"""
        
        message += """
✅ <b>Phân tích hoàn thành!</b>
📱 Bot đang hoạt động bình thường
"""
        
        # Gửi về Telegram
        success = send_telegram_message(message)
        
        if success:
            logger.info("🎉 Phân tích và gửi Telegram thành công!")
        else:
            logger.error("❌ Lỗi gửi Telegram")
    else:
        logger.error("❌ Không có kết quả phân tích")
    
    logger.info("🏁 Hoàn thành!")

if __name__ == "__main__":
    main()
