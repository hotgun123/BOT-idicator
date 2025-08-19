#!/usr/bin/env python3
"""
Script test có ML - tối ưu để chạy nhanh trên GitHub Actions
Có ML predictions nhưng không train models mới
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

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

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

def create_ml_features(data):
    """Tạo features cho ML (đơn giản hóa)"""
    try:
        df = pd.DataFrame(data)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Technical indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
        df['bb_upper'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
        df['bb_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Price features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Target (1 if price goes up next period, 0 if down)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Remove NaN values
        df = df.dropna()
        
        # Feature columns
        feature_columns = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 
                          'ema_20', 'ema_50', 'sma_20', 'atr', 'price_change', 
                          'high_low_ratio', 'volume_ratio']
        
        return df, feature_columns
        
    except Exception as e:
        logger.error(f"❌ Lỗi tạo features: {e}")
        return None, None

def train_simple_ml_model(df, feature_columns):
    """Train ML model đơn giản"""
    try:
        if len(df) < 50:  # Cần ít nhất 50 mẫu
            logger.warning("⚠️ Không đủ dữ liệu để train ML model")
            return None, None
        
        X = df[feature_columns]
        y = df['target']
        
        # Train test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        accuracy = model.score(X_test_scaled, y_test)
        logger.info(f"✅ ML Model trained với accuracy: {accuracy:.3f}")
        
        return model, scaler
        
    except Exception as e:
        logger.error(f"❌ Lỗi train ML model: {e}")
        return None, None

def predict_with_ml(model, scaler, current_features):
    """Dự đoán với ML model"""
    try:
        if model is None or scaler is None:
            return None, 0.0
        
        # Scale features
        features_scaled = scaler.transform([current_features])
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0].max()
        
        return prediction, confidence
        
    except Exception as e:
        logger.error(f"❌ Lỗi dự đoán ML: {e}")
        return None, 0.0

def analyze_symbol_with_ml(symbol):
    """Phân tích symbol với ML"""
    try:
        logger.info(f"🔍 Đang phân tích {symbol} với ML...")
        
        # Lấy dữ liệu
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=200)
        if not ohlcv:
            logger.warning(f"⚠️ Không lấy được dữ liệu cho {symbol}")
            return None
        
        # Tạo features
        df, feature_columns = create_ml_features(ohlcv)
        if df is None:
            return None
        
        # Train ML model
        model, scaler = train_simple_ml_model(df, feature_columns)
        
        # Lấy features hiện tại
        current_features = df[feature_columns].iloc[-1].values
        
        # Dự đoán ML
        ml_prediction, ml_confidence = predict_with_ml(model, scaler, current_features)
        
        # Technical analysis
        current_price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        ema_20 = df['ema_20'].iloc[-1]
        ema_50 = df['ema_50'].iloc[-1]
        
        # Tín hiệu kết hợp
        ta_signal = "NEUTRAL"
        if rsi < 30 and macd > macd_signal and current_price > ema_20:
            ta_signal = "LONG"
        elif rsi > 70 and macd < macd_signal and current_price < ema_20:
            ta_signal = "SHORT"
        
        # Kết hợp TA và ML
        final_signal = ta_signal
        if ml_prediction is not None:
            ml_signal = "LONG" if ml_prediction == 1 else "SHORT"
            if ml_confidence > 0.6:  # Chỉ tin tưởng ML nếu confidence > 60%
                final_signal = ml_signal
        
        return {
            'symbol': symbol,
            'price': current_price,
            'rsi': rsi,
            'macd': macd,
            'ta_signal': ta_signal,
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence,
            'ml_signal': "LONG" if ml_prediction == 1 else "SHORT" if ml_prediction == 0 else "N/A",
            'final_signal': final_signal,
            'ema_20': ema_20,
            'ema_50': ema_50
        }
        
    except Exception as e:
        logger.error(f"❌ Lỗi phân tích {symbol}: {e}")
        return None

def main():
    """Main function với ML"""
    logger.info("🚀 Bắt đầu phân tích với ML...")
    
    # Phân tích BTC và ETH
    symbols = ['BTC/USDT', 'ETH/USDT']
    results = []
    
    for symbol in symbols:
        result = analyze_symbol_with_ml(symbol)
        if result:
            results.append(result)
            logger.info(f"✅ Đã phân tích {symbol} với ML")
    
    if results:
        # Tạo message
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"""
🤖 <b>PHÂN TÍCH TRADING VỚI ML</b>

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

🎯 <b>TÍN HIỆU:</b>
• Technical Analysis: <b>{result['ta_signal']}</b>
• Machine Learning: <b>{result['ml_signal']}</b> (Confidence: {result['ml_confidence']:.1%})
• <b>Kết luận cuối: {result['final_signal']}</b>

📉 EMA20: ${result['ema_20']:,.2f}
📉 EMA50: ${result['ema_50']:,.2f}

"""
        
        message += """
✅ <b>Phân tích hoàn thành!</b>
🤖 ML Model đã được train và dự đoán
📱 Bot đang hoạt động bình thường
"""
        
        # Gửi về Telegram
        success = send_telegram_message(message)
        
        if success:
            logger.info("🎉 Phân tích ML và gửi Telegram thành công!")
        else:
            logger.error("❌ Lỗi gửi Telegram")
    else:
        logger.error("❌ Không có kết quả phân tích")
    
    logger.info("🏁 Hoàn thành!")

if __name__ == "__main__":
    main()
