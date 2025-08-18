import ccxt
import talib
import numpy as np
import logging
from datetime import datetime
import time
import requests
import threading
import yfinance as yf
import pandas as pd
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo kết nối với Binance mainnet (spot)
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
    }
})

# Khởi tạo kết nối với Exness cho hàng hóa (tạm thời comment lại)
# exness_exchange = ccxt.exness({
#     'enableRateLimit': True,
#     'options': {
#         'defaultType': 'spot',
#         'adjustForTimeDifference': True,
#     }
# })
exness_exchange = None  # Tạm thời không sử dụng Exness

# Cấu hình cho TradingView và Investing.com
TRADINGVIEW_SYMBOLS = {
    'XAU/USD': 'XAUUSD',  # Vàng
    'WTI/USD': 'USOIL'    # Dầu WTI
}

INVESTING_SYMBOLS = {
    'XAU/USD': 'gold',    # Vàng trên Investing.com
    'WTI/USD': 'wti-crude-oil'  # Dầu WTI trên Investing.com
}

# Cấu hình cho Exness
EXNESS_SYMBOLS = {
    'XAU/USD': 'XAUUSD',  # Vàng trên Exness
    'WTI/USD': 'WTIUSD'   # Dầu WTI trên Exness
}

# Cấu hình
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XAU/USD', 'WTI/USD']
TIMEFRAMES = ['1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
CANDLE_LIMIT = 200
SIGNAL_THRESHOLD = 0.6 # Giảm xuống 40% để dễ có tín hiệu hơn
RETRY_ATTEMPTS = 2

# Cấu hình Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7496162935:AAGncIsO4q18cOWRGpK0vYb_5zWxYNEgWKQ")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1866335373")
TELEGRAM_REPORT_INTERVAL = 7200  # 2 tiếng = 7200 giây

# Cấu hình cho hệ thống theo dõi dự đoán
PREDICTION_DATA_DIR = "prediction_data"
PREDICTION_HISTORY_FILE = "prediction_history.json"
PREDICTION_ACCURACY_FILE = "prediction_accuracy.json"
PREDICTION_UPDATE_INTERVAL = 3600  # Cập nhật kết quả thực tế mỗi giờ
PREDICTION_RETENTION_DAYS = 30  # Giữ dữ liệu dự đoán trong 30 ngày

def get_usdt_symbols():
    """Trả về danh sách cặp giao dịch cố định bao gồm crypto, vàng và dầu"""
    return SYMBOLS

def ensure_prediction_data_dir():
    """Đảm bảo thư mục dữ liệu dự đoán tồn tại"""
    Path(PREDICTION_DATA_DIR).mkdir(exist_ok=True)

def save_prediction(symbol, timeframe, prediction_data, current_price):
    """Lưu dự đoán mới vào hệ thống"""
    try:
        ensure_prediction_data_dir()
        
        prediction = {
            'id': f"{symbol}_{timeframe}_{int(time.time())}",
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'prediction': {
                'trend': prediction_data.get('trend', 'neutral'),
                'signal': prediction_data.get('signal', 'Hold'),
                'confidence': prediction_data.get('confidence', 0),
                'entry_price': prediction_data.get('entry_points', {}).get('entry_price', current_price),
                'stop_loss': prediction_data.get('entry_points', {}).get('stop_loss', 0),
                'take_profit': prediction_data.get('entry_points', {}).get('take_profit', 0),
                'risk_reward_ratio': prediction_data.get('entry_points', {}).get('risk_reward_ratio', 0),
                'analysis_summary': prediction_data.get('analysis_summary', ''),
                'technical_signals': prediction_data.get('technical_signals', {}),
                'commodity_signals': prediction_data.get('commodity_signals', {}),
                'smc_signals': prediction_data.get('smc_signals', {}),
                'price_action_signals': prediction_data.get('price_action_signals', {})
            },
            'actual_result': None,
            'accuracy': None,
            'status': 'pending'
        }
        
        # Đọc dữ liệu hiện tại
        history_file = os.path.join(PREDICTION_DATA_DIR, PREDICTION_HISTORY_FILE)
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        
        # Thêm dự đoán mới
        history.append(prediction)
        
        # Lưu lại
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Đã lưu dự đoán cho {symbol} ({timeframe}): {prediction['prediction']['signal']}")
        return prediction['id']
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lưu dự đoán: {e}")
        return None

def update_prediction_results():
    """Cập nhật kết quả thực tế cho các dự đoán đang chờ"""
    try:
        ensure_prediction_data_dir()
        history_file = os.path.join(PREDICTION_DATA_DIR, PREDICTION_HISTORY_FILE)
        
        if not os.path.exists(history_file):
            return
        
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        updated = False
        current_time = datetime.now()
        
        for prediction in history:
            if prediction['status'] != 'pending':
                continue
            
            # Kiểm tra xem đã đến lúc cập nhật chưa (dựa trên timeframe)
            prediction_time = datetime.fromisoformat(prediction['timestamp'])
            timeframe = prediction['timeframe']
            
            # Tính thời gian cần thiết để đánh giá dự đoán
            evaluation_time = get_evaluation_time(timeframe)
            if (current_time - prediction_time).total_seconds() < evaluation_time:
                continue
            
            # Lấy giá hiện tại
            current_price = get_current_price_for_prediction(prediction['symbol'])
            if current_price is None:
                continue
            
            # Tính toán kết quả thực tế
            actual_result = calculate_actual_result(
                prediction['prediction'],
                prediction['current_price'],
                current_price,
                timeframe
            )
            
            # Cập nhật dự đoán
            prediction['actual_result'] = actual_result
            prediction['accuracy'] = calculate_prediction_accuracy(
                prediction['prediction'],
                actual_result
            )
            prediction['status'] = 'completed'
            updated = True
            
            logger.info(f"📊 Cập nhật kết quả {prediction['symbol']}: {actual_result['outcome']} (Độ chính xác: {prediction['accuracy']:.1%})")
        
        if updated:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
            # Cập nhật thống kê độ chính xác
            update_accuracy_statistics()
    
    except Exception as e:
        logger.error(f"❌ Lỗi khi cập nhật kết quả dự đoán: {e}")

def get_evaluation_time(timeframe):
    """Tính thời gian cần thiết để đánh giá dự đoán dựa trên timeframe"""
    timeframe_hours = {
        '1h': 2,    # Đánh giá sau 2 giờ
        '2h': 4,    # Đánh giá sau 4 giờ
        '4h': 8,    # Đánh giá sau 8 giờ
        '6h': 12,   # Đánh giá sau 12 giờ
        '8h': 16,   # Đánh giá sau 16 giờ
        '12h': 24,  # Đánh giá sau 24 giờ
        '1d': 48,   # Đánh giá sau 48 giờ
        '3d': 144,  # Đánh giá sau 6 ngày
        '1w': 336   # Đánh giá sau 14 ngày
    }
    return timeframe_hours.get(timeframe, 24) * 3600  # Chuyển sang giây

def get_current_price_for_prediction(symbol):
    """Lấy giá hiện tại cho việc cập nhật dự đoán"""
    try:
        if symbol in ['XAU/USD', 'WTI/USD']:
            return get_commodity_current_price(symbol)
        else:
            ticker = exchange.fetch_ticker(symbol)
            return ticker['last']
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy giá hiện tại cho {symbol}: {e}")
        return None

def calculate_actual_result(prediction, initial_price, current_price, timeframe):
    """Tính toán kết quả thực tế của dự đoán"""
    price_change = (current_price - initial_price) / initial_price
    price_change_percent = price_change * 100
    
    # Lấy thông tin dự đoán
    predicted_trend = prediction.get('trend', 'neutral')
    predicted_signal = prediction.get('signal', 'Hold')
    entry_price = prediction.get('entry_price', initial_price)
    stop_loss = prediction.get('stop_loss', 0)
    take_profit = prediction.get('take_profit', 0)
    
    # Xác định kết quả
    if predicted_signal == 'Long':
        if current_price > entry_price:
            if take_profit > 0 and current_price >= take_profit:
                outcome = 'take_profit_hit'
            else:
                outcome = 'profitable'
        elif stop_loss > 0 and current_price <= stop_loss:
            outcome = 'stop_loss_hit'
        else:
            outcome = 'loss'
    elif predicted_signal == 'Short':
        if current_price < entry_price:
            if take_profit > 0 and current_price <= take_profit:
                outcome = 'take_profit_hit'
            else:
                outcome = 'profitable'
        elif stop_loss > 0 and current_price >= stop_loss:
            outcome = 'stop_loss_hit'
        else:
            outcome = 'loss'
    else:  # Hold
        if abs(price_change) < 0.01:  # Thay đổi < 1%
            outcome = 'correct_hold'
        else:
            outcome = 'missed_opportunity'
    
    return {
        'current_price': current_price,
        'price_change': price_change,
        'price_change_percent': price_change_percent,
        'outcome': outcome,
        'profit_loss': current_price - entry_price if entry_price > 0 else 0,
        'profit_loss_percent': ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
    }

def calculate_prediction_accuracy(prediction, actual_result):
    """Tính toán độ chính xác của dự đoán"""
    predicted_signal = prediction.get('signal', 'Hold')
    outcome = actual_result.get('outcome', 'unknown')
    
    # Định nghĩa độ chính xác
    accuracy_map = {
        'Long': {
            'profitable': 1.0,
            'take_profit_hit': 1.0,
            'stop_loss_hit': 0.0,
            'loss': 0.0
        },
        'Short': {
            'profitable': 1.0,
            'take_profit_hit': 1.0,
            'stop_loss_hit': 0.0,
            'loss': 0.0
        },
        'Hold': {
            'correct_hold': 1.0,
            'missed_opportunity': 0.5
        }
    }
    
    return accuracy_map.get(predicted_signal, {}).get(outcome, 0.0)

def update_accuracy_statistics():
    """Cập nhật thống kê độ chính xác tổng thể"""
    try:
        history_file = os.path.join(PREDICTION_DATA_DIR, PREDICTION_HISTORY_FILE)
        accuracy_file = os.path.join(PREDICTION_DATA_DIR, PREDICTION_ACCURACY_FILE)
        
        if not os.path.exists(history_file):
            return
        
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # Lọc các dự đoán đã hoàn thành
        completed_predictions = [p for p in history if p['status'] == 'completed']
        
        if not completed_predictions:
            return
        
        # Tính toán thống kê
        total_predictions = len(completed_predictions)
        accurate_predictions = len([p for p in completed_predictions if p['accuracy'] >= 0.8])
        overall_accuracy = accurate_predictions / total_predictions if total_predictions > 0 else 0
        
        # Thống kê theo symbol
        symbol_stats = {}
        for prediction in completed_predictions:
            symbol = prediction['symbol']
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {'total': 0, 'accurate': 0, 'accuracy': 0}
            
            symbol_stats[symbol]['total'] += 1
            if prediction['accuracy'] >= 0.8:
                symbol_stats[symbol]['accurate'] += 1
        
        # Tính độ chính xác cho từng symbol
        for symbol in symbol_stats:
            stats = symbol_stats[symbol]
            stats['accuracy'] = stats['accurate'] / stats['total'] if stats['total'] > 0 else 0
        
        # Thống kê theo timeframe
        timeframe_stats = {}
        for prediction in completed_predictions:
            timeframe = prediction['timeframe']
            if timeframe not in timeframe_stats:
                timeframe_stats[timeframe] = {'total': 0, 'accurate': 0, 'accuracy': 0}
            
            timeframe_stats[timeframe]['total'] += 1
            if prediction['accuracy'] >= 0.8:
                timeframe_stats[timeframe]['accurate'] += 1
        
        # Tính độ chính xác cho từng timeframe
        for timeframe in timeframe_stats:
            stats = timeframe_stats[timeframe]
            stats['accuracy'] = stats['accurate'] / stats['total'] if stats['total'] > 0 else 0
        
        # Lưu thống kê
        accuracy_data = {
            'last_updated': datetime.now().isoformat(),
            'overall': {
                'total_predictions': total_predictions,
                'accurate_predictions': accurate_predictions,
                'accuracy': overall_accuracy
            },
            'by_symbol': symbol_stats,
            'by_timeframe': timeframe_stats
        }
        
        with open(accuracy_file, 'w', encoding='utf-8') as f:
            json.dump(accuracy_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📈 Đã cập nhật thống kê độ chính xác: {overall_accuracy:.1%} ({accurate_predictions}/{total_predictions})")
    
    except Exception as e:
        logger.error(f"❌ Lỗi khi cập nhật thống kê độ chính xác: {e}")

def get_prediction_accuracy_data():
    """Lấy dữ liệu độ chính xác dự đoán"""
    try:
        accuracy_file = os.path.join(PREDICTION_DATA_DIR, PREDICTION_ACCURACY_FILE)
        if os.path.exists(accuracy_file):
            with open(accuracy_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"❌ Lỗi khi đọc dữ liệu độ chính xác: {e}")
        return None

def cleanup_old_predictions():
    """Dọn dẹp các dự đoán cũ"""
    try:
        history_file = os.path.join(PREDICTION_DATA_DIR, PREDICTION_HISTORY_FILE)
        if not os.path.exists(history_file):
            return
        
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        current_time = datetime.now()
        cutoff_time = current_time.timestamp() - (PREDICTION_RETENTION_DAYS * 24 * 3600)
        
        # Lọc bỏ các dự đoán cũ
        filtered_history = []
        removed_count = 0
        
        for prediction in history:
            prediction_timestamp = datetime.fromisoformat(prediction['timestamp']).timestamp()
            if prediction_timestamp > cutoff_time:
                filtered_history.append(prediction)
            else:
                removed_count += 1
        
        if removed_count > 0:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_history, f, ensure_ascii=False, indent=2)
            
            logger.info(f"🧹 Đã dọn dẹp {removed_count} dự đoán cũ")
    
    except Exception as e:
        logger.error(f"❌ Lỗi khi dọn dẹp dự đoán cũ: {e}")

def adjust_analysis_based_on_accuracy(analysis_result, symbol, timeframe):
    """Điều chỉnh phân tích dựa trên độ chính xác lịch sử"""
    try:
        accuracy_data = get_prediction_accuracy_data()
        if not accuracy_data:
            return analysis_result
        
        # Lấy độ chính xác cho symbol và timeframe
        symbol_accuracy = accuracy_data.get('by_symbol', {}).get(symbol, {}).get('accuracy', 0.5)
        timeframe_accuracy = accuracy_data.get('by_timeframe', {}).get(timeframe, {}).get('accuracy', 0.5)
        
        # Tính độ chính xác trung bình
        avg_accuracy = (symbol_accuracy + timeframe_accuracy) / 2
        
        # Điều chỉnh confidence dựa trên độ chính xác
        current_confidence = analysis_result.get('confidence', 0.5)
        
        if avg_accuracy > 0.7:  # Độ chính xác cao
            adjusted_confidence = min(current_confidence * 1.2, 1.0)
            analysis_result['confidence'] = adjusted_confidence
            analysis_result['accuracy_adjustment'] = f"Tăng confidence do độ chính xác cao ({avg_accuracy:.1%})"
        elif avg_accuracy < 0.4:  # Độ chính xác thấp
            adjusted_confidence = max(current_confidence * 0.8, 0.1)
            analysis_result['confidence'] = adjusted_confidence
            analysis_result['accuracy_adjustment'] = f"Giảm confidence do độ chính xác thấp ({avg_accuracy:.1%})"
        else:
            analysis_result['accuracy_adjustment'] = f"Độ chính xác trung bình ({avg_accuracy:.1%})"
        
        # Thêm thông tin độ chính xác vào analysis
        analysis_result['historical_accuracy'] = {
            'symbol_accuracy': symbol_accuracy,
            'timeframe_accuracy': timeframe_accuracy,
            'average_accuracy': avg_accuracy
        }
        
        return analysis_result
    
    except Exception as e:
        logger.error(f"❌ Lỗi khi điều chỉnh phân tích: {e}")
        return analysis_result

def send_prediction_accuracy_report():
    """Gửi báo cáo độ chính xác dự đoán qua Telegram"""
    try:
        accuracy_report = format_prediction_accuracy_report()
        if send_telegram_message(accuracy_report):
            logger.info("📊 Đã gửi báo cáo độ chính xác dự đoán qua Telegram")
            return True
        else:
            logger.error("❌ Không thể gửi báo cáo độ chính xác dự đoán")
            return False
    except Exception as e:
        logger.error(f"❌ Lỗi khi gửi báo cáo độ chính xác: {e}")
        return False

def fetch_commodity_data(symbol, timeframe, limit):
    """Lấy dữ liệu hàng hóa từ Yahoo Finance, TradingView và Investing.com"""
    try:
        # Thử Yahoo Finance trước (ưu tiên cao nhất)
        yf_data = fetch_yahoo_finance_data(symbol, timeframe, limit)
        if yf_data:
            logger.info(f"✅ Lấy dữ liệu {symbol} từ Yahoo Finance thành công")
            return yf_data
        
        # Thử TradingView nếu Yahoo Finance thất bại
        tv_data = fetch_tradingview_data(symbol, timeframe, limit)
        if tv_data:
            logger.info(f"✅ Lấy dữ liệu {symbol} từ TradingView thành công")
            return tv_data
        
        # Nếu TradingView thất bại, thử Investing.com
        investing_data = fetch_investing_data(symbol, timeframe, limit)
        if investing_data:
            logger.info(f"✅ Lấy dữ liệu {symbol} từ Investing.com thành công")
            return investing_data
        
        logger.error(f"❌ Không thể lấy dữ liệu cho {symbol} từ bất kỳ nguồn nào")
        return None
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy dữ liệu hàng hóa cho {symbol}: {e}")
        return None



def fetch_tradingview_data(symbol, timeframe, limit):
    """Lấy dữ liệu từ TradingView"""
    try:
        tv_symbol = TRADINGVIEW_SYMBOLS.get(symbol)
        if not tv_symbol:
            return None
        
        # Chuyển đổi timeframe
        interval_map = {
            '1h': '1',
            '2h': '2', 
            '4h': '4',
            '6h': '6',
            '8h': '8',
            '12h': '12',
            '1d': '1D',
            '3d': '3D',
            '1w': '1W'
        }
        
        interval = interval_map.get(timeframe, '1D')
        
        # Sử dụng TradingView API (cần cài đặt tradingview-ta)
        try:
            from tradingview_ta import TA_Handler, Interval
            handler = TA_Handler(
                symbol=tv_symbol,
                exchange="OANDA",
                screener="forex",
                interval=interval,
                timeout=10
            )
            
            # Lấy dữ liệu OHLCV
            analysis = handler.get_analysis()
            if analysis and hasattr(analysis, 'indicators'):
                # Tạo dữ liệu giả lập từ indicators
                # TradingView API chỉ trả về indicators, không phải OHLCV
                # Nên chúng ta sẽ sử dụng Yahoo Finance thay thế
                return None
                
        except ImportError:
            logger.warning("TradingView TA library chưa được cài đặt")
            return None
            
    except Exception as e:
        logger.error(f"Lỗi khi lấy dữ liệu TradingView cho {symbol}: {e}")
        return None

def fetch_investing_data(symbol, timeframe, limit):
    """Lấy dữ liệu từ Investing.com"""
    try:
        investing_symbol = INVESTING_SYMBOLS.get(symbol)
        if not investing_symbol:
            return None
        
        # Investing.com không có API công khai, nên chúng ta sẽ sử dụng Yahoo Finance
        # hoặc web scraping (cần thêm thư viện)
        return None
        
    except Exception as e:
        logger.error(f"Lỗi khi lấy dữ liệu Investing.com cho {symbol}: {e}")
        return None

def fetch_yahoo_finance_data(symbol, timeframe, limit):
    """Lấy dữ liệu từ Yahoo Finance (fallback)"""
    try:
        # Map symbols cho Yahoo Finance
        yf_symbols = {
            'XAU/USD': 'GC=F',  # Gold Futures
            'WTI/USD': 'CL=F'   # Crude Oil Futures
        }
        
        yf_symbol = yf_symbols.get(symbol)
        if not yf_symbol:
            return None
        
        # Chuyển đổi timeframe - sử dụng các interval được Yahoo Finance hỗ trợ
        period_map = {
            '1h': '5d',
            '2h': '5d', 
            '4h': '5d',
            '6h': '5d',
            '8h': '5d',
            '12h': '5d',
            '1d': '1mo',
            '3d': '3mo',
            '1w': '6mo'
        }
        
        interval_map = {
            '1h': '1h',
            '2h': '1h',  # Yahoo Finance không hỗ trợ 2h, dùng 1h
            '4h': '1h',  # Yahoo Finance không hỗ trợ 4h, dùng 1h
            '6h': '1h',  # Yahoo Finance không hỗ trợ 6h, dùng 1h
            '8h': '1h',  # Yahoo Finance không hỗ trợ 8h, dùng 1h
            '12h': '1h', # Yahoo Finance không hỗ trợ 12h, dùng 1h
            '1d': '1d',
            '3d': '1d',  # Yahoo Finance không hỗ trợ 3d, dùng 1d
            '1w': '1wk'
        }
        
        period = period_map.get(timeframe, '1mo')
        interval = interval_map.get(timeframe, '1d')
        
        # Lấy dữ liệu từ Yahoo Finance
        ticker = yf.Ticker(yf_symbol)
        data = ticker.history(period=period, interval=interval)
        
        logger.info(f"📊 Yahoo Finance: Lấy {len(data)} dòng dữ liệu cho {symbol} ({period}, {interval})")
        
        if data.empty:
            logger.warning(f"Không có dữ liệu cho {symbol} từ Yahoo Finance")
            return None
        
        # Chuyển đổi sang format OHLCV
        ohlcv = []
        for index, row in data.tail(limit).iterrows():
            ohlcv.append({
                'timestamp': int(index.timestamp() * 1000),
                'open': float(row['Open']),
                'high': float(row['High']), 
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume']) if 'Volume' in row and not np.isnan(row['Volume']) else 1000000.0  # Volume mặc định
            })
        
        return ohlcv
        
    except Exception as e:
        logger.error(f"Lỗi khi lấy dữ liệu Yahoo Finance cho {symbol}: {e}")
        return None

def get_commodity_current_price(symbol):
    """Lấy giá hiện tại cho hàng hóa từ Yahoo Finance"""
    try:
        # Sử dụng Yahoo Finance
        yf_symbols = {
            'XAU/USD': 'GC=F',  # Gold Futures
            'WTI/USD': 'CL=F'   # Crude Oil Futures
        }
        
        yf_symbol = yf_symbols.get(symbol)
        if not yf_symbol:
            return None
        
        # Lấy thông tin ticker
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info
        
        # Lấy giá hiện tại
        current_price = info.get('regularMarketPrice')
        if current_price:
            logger.info(f"✅ Thành công lấy giá {symbol} từ Yahoo Finance: ${current_price}")
            return current_price
        
        # Fallback: lấy từ lịch sử gần nhất
        data = ticker.history(period='1d')
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            logger.info(f"✅ Thành công lấy giá {symbol} từ lịch sử Yahoo Finance: ${current_price}")
            return current_price
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy giá hiện tại cho {symbol}: {e}")
        return None

def fetch_ohlcv(symbol, timeframe, limit):
    """Lấy dữ liệu OHLCV cho crypto, vàng và dầu"""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            # Xử lý đặc biệt cho vàng và dầu - sử dụng Yahoo Finance/TradingView/Investing
            if symbol in ['XAU/USD', 'WTI/USD']:
                ohlcv = fetch_commodity_data(symbol, timeframe, limit)
                logger.info(f"🔍 Commodity data for {symbol}: {len(ohlcv) if ohlcv else 0} candles, need {limit * 0.8}")
                if ohlcv and len(ohlcv) >= limit * 0.8:
                    logger.info(f"✅ Thành công lấy dữ liệu {symbol} từ Yahoo Finance/TradingView/Investing")
                    return {
                        'open': np.array([candle['open'] for candle in ohlcv]),
                        'high': np.array([candle['high'] for candle in ohlcv]),
                        'low': np.array([candle['low'] for candle in ohlcv]),
                        'close': np.array([candle['close'] for candle in ohlcv]),
                        'volume': np.array([candle['volume'] for candle in ohlcv])
                    }
                else:
                    logger.warning(f"⚠️ Không thể lấy dữ liệu cho {symbol} từ Yahoo Finance/TradingView/Investing")
                    return None
            else:
                # Xử lý bình thường cho crypto
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if len(ohlcv) < limit * 0.8:
                logger.warning(f"⚠️ Dữ liệu OHLCV cho {symbol} ({timeframe}) không đủ: {len(ohlcv)}/{limit}")
                return None
                
            return {
                'open': np.array([candle[1] for candle in ohlcv]),
                'high': np.array([candle[2] for candle in ohlcv]),
                'low': np.array([candle[3] for candle in ohlcv]),
                'close': np.array([candle[4] for candle in ohlcv]),
                'volume': np.array([candle[5] for candle in ohlcv])
            }
        except Exception as e:
            error_msg = str(e)
            if "Invalid symbol status" in error_msg or "symbol not found" in error_msg.lower():
                logger.warning(f"⚠️ Symbol {symbol} không khả dụng cho {timeframe}: {error_msg}")
                return None
            elif attempt < RETRY_ATTEMPTS - 1:
                logger.warning(f"⚠️ Lỗi khi lấy dữ liệu OHLCV cho {symbol} ({timeframe}, lần {attempt + 1}/{RETRY_ATTEMPTS}): {error_msg}")
                time.sleep(1)
            else:
                logger.error(f"❌ Không thể lấy dữ liệu OHLCV cho {symbol} ({timeframe}): {error_msg}")
    return None

def calculate_fibonacci_levels(highs, lows):
    """Tính các mức Fibonacci Retracement"""
    max_price = max(highs[-50:])
    min_price = min(lows[-50:])
    diff = max_price - min_price
    levels = {
        '23.6%': max_price - 0.236 * diff,
        '38.2%': max_price - 0.382 * diff,
        '50%': max_price - 0.5 * diff,
        '61.8%': max_price - 0.618 * diff
    }
    return levels

def find_support_resistance(highs, lows, current_price):
    """Tìm mức hỗ trợ/kháng cự gần nhất"""
    fib_levels = calculate_fibonacci_levels(highs, lows)
    support = min([price for price in fib_levels.values() if price < current_price], default=min(lows[-20:]))
    resistance = max([price for price in fib_levels.values() if price > current_price], default=max(highs[-20:]))
    return support, resistance

def calculate_pivot_points(highs, lows, closes):
    """Tính các mức Pivot Points (Classic)"""
    high = highs[-1]
    low = lows[-1]
    close = closes[-1]
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return {'pivot': pivot, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2, 'r3': r3, 's3': s3}

def calculate_volume_profile(highs, lows, volumes, bins=10):
    """Tính Volume Profile đơn giản (phân bố khối lượng theo mức giá)"""
    price_range = max(highs[-50:]) - min(lows[-50:])
    bin_size = price_range / bins
    volume_bins = [0] * bins
    for i in range(len(highs[-50:])):
        price = (highs[-50:][i] + lows[-50:][i]) / 2
        bin_index = min(int((price - min(lows[-50:])) / bin_size), bins - 1)
        volume_bins[bin_index] += volumes[-50:][i]
    max_volume_price = min(lows[-50:]) + volume_bins.index(max(volume_bins)) * bin_size
    return max_volume_price

def calculate_vwap(highs, lows, closes, volumes):
    """Tính VWAP (Volume Weighted Average Price)"""
    typical_prices = (highs[-20:] + lows[-20:] + closes[-20:]) / 3
    vwap = np.sum(typical_prices * volumes[-20:]) / np.sum(volumes[-20:])
    return vwap

def detect_price_patterns(highs, lows, closes):
    """Phát hiện các mô hình giá"""
    pattern = 'None'
    
    # Đỉnh đầu vai (Head and Shoulders)
    if len(highs) >= 7:
        left_shoulder = highs[-5] > highs[-6] and highs[-5] > highs[-4]
        head = highs[-3] > highs[-5] and highs[-3] > highs[-1]
        right_shoulder = highs[-1] > highs[-2] and highs[-1] < highs[-3]
        if left_shoulder and head and right_shoulder:
            pattern = 'Head and Shoulders'
    
    # Đỉnh đôi (Double Top)
    elif len(highs) >= 5:
        if abs(highs[-3] - highs[-1]) / highs[-3] < 0.01 and highs[-3] > highs[-2] and highs[-1] > highs[-2]:
            pattern = 'Double Top'
    
    # Cờ (Flag)
    elif len(highs) >= 10:
        uptrend = all(closes[i] > closes[i-1] for i in range(-10, -5))
        consolidation = max(highs[-5:]) - min(lows[-5:]) < 0.02 * closes[-1]
        if uptrend and consolidation:
            pattern = 'Flag'
    
    return pattern

def interpret_candlestick_patterns(patterns, current_price, ema50):
    """Phân tích ý nghĩa của các mô hình nến và đưa ra kết luận cụ thể"""
    bullish_patterns = ['Hammer', 'Bullish Engulfing', 'Morning Star', 'Three White Soldiers']
    bearish_patterns = ['Shooting Star', 'Bearish Engulfing', 'Evening Star', 'Three Black Crows']
    neutral_patterns = ['Doji', 'Spinning Top']
    
    bullish_count = sum(1 for p in patterns if p in bullish_patterns)
    bearish_count = sum(1 for p in patterns if p in bearish_patterns)
    neutral_count = sum(1 for p in patterns if p in neutral_patterns)
    
    # Phân tích chi tiết từng mô hình
    analysis = []
    
    for pattern in patterns:
        if pattern == 'Doji':
            if current_price > ema50:
                analysis.append("Doji ở vùng kháng cự → Cảnh báo đảo chiều giảm")
            else:
                analysis.append("Doji ở vùng hỗ trợ → Cảnh báo đảo chiều tăng")
        elif pattern == 'Hammer':
            analysis.append("Hammer → Tín hiệu đảo chiều tăng mạnh")
        elif pattern == 'Shooting Star':
            analysis.append("Shooting Star → Tín hiệu đảo chiều giảm mạnh")
        elif pattern == 'Bullish Engulfing':
            analysis.append("Bullish Engulfing → Tín hiệu tăng mạnh, đảo chiều từ giảm")
        elif pattern == 'Bearish Engulfing':
            analysis.append("Bearish Engulfing → Tín hiệu giảm mạnh, đảo chiều từ tăng")
        elif pattern == 'Morning Star':
            analysis.append("Morning Star → Tín hiệu đảo chiều tăng rất mạnh")
        elif pattern == 'Evening Star':
            analysis.append("Evening Star → Tín hiệu đảo chiều giảm rất mạnh")
        elif pattern == 'Three White Soldiers':
            analysis.append("Three White Soldiers → Xu hướng tăng mạnh tiếp diễn")
        elif pattern == 'Three Black Crows':
            analysis.append("Three Black Crows → Xu hướng giảm mạnh tiếp diễn")
        elif pattern == 'Spinning Top':
            analysis.append("Spinning Top → Lưỡng lự, cần xác nhận thêm")
    
    # Kết luận tổng thể
    if bullish_count > bearish_count and bullish_count > 0:
        conclusion = f"🟢 TÍN HIỆU TĂNG: {bullish_count} mô hình bullish vs {bearish_count} bearish"
    elif bearish_count > bullish_count and bearish_count > 0:
        conclusion = f"🔴 TÍN HIỆU GIẢM: {bearish_count} mô hình bearish vs {bullish_count} bullish"
    elif neutral_count > 0 and bullish_count == 0 and bearish_count == 0:
        conclusion = f"🟡 LƯỠNG LỰ: {neutral_count} mô hình trung tính"
    else:
        conclusion = "⚪ KHÔNG CÓ MÔ HÌNH NẾN RÕ RÀNG"
    
    return {
        'patterns': patterns,
        'analysis': analysis,
        'conclusion': conclusion,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'neutral_count': neutral_count
    }

def detect_candlestick_patterns(opens, highs, lows, closes):
    """Phát hiện các mô hình nến Nhật"""
    patterns = []
    
    # Nến đơn
    body_size = abs(opens[-1] - closes[-1])
    candle_range = highs[-1] - lows[-1]
    if candle_range > 0:
        # Doji
        if body_size / candle_range < 0.1:
            patterns.append('Doji')
        # Hammer
        if (closes[-1] > opens[-1] and 
            (highs[-1] - closes[-1]) / candle_range < 0.2 and 
            (opens[-1] - lows[-1]) / candle_range > 0.6):
            patterns.append('Hammer')
        # Shooting Star
        if (closes[-1] < opens[-1] and 
            (highs[-1] - opens[-1]) / candle_range > 0.6 and 
            (closes[-1] - lows[-1]) / candle_range < 0.2):
            patterns.append('Shooting Star')
        # Spinning Top
        if body_size / candle_range < 0.3 and (highs[-1] - closes[-1]) / candle_range > 0.3 and (opens[-1] - lows[-1]) / candle_range > 0.3:
            patterns.append('Spinning Top')

    # Nến đôi
    if len(opens) >= 2:
        # Bullish Engulfing
        if (closes[-2] < opens[-2] and closes[-1] > opens[-1] and 
            closes[-1] > opens[-2] and opens[-1] < closes[-2]):
            patterns.append('Bullish Engulfing')
        # Bearish Engulfing
        if (closes[-2] > opens[-2] and closes[-1] < opens[-1] and 
            closes[-1] < opens[-2] and opens[-1] > closes[-2]):
            patterns.append('Bearish Engulfing')

    # Nến ba
    if len(opens) >= 3:
        # Morning Star
        if (closes[-3] < opens[-3] and 
            abs(closes[-2] - opens[-2]) / (highs[-2] - lows[-2]) < 0.3 and 
            closes[-1] > opens[-1] and closes[-1] > (highs[-3] + lows[-3]) / 2):
            patterns.append('Morning Star')
        # Evening Star
        if (closes[-3] > opens[-3] and 
            abs(closes[-2] - opens[-2]) / (highs[-2] - lows[-2]) < 0.3 and 
            closes[-1] < opens[-1] and closes[-1] < (highs[-3] + lows[-3]) / 2):
            patterns.append('Evening Star')
        # Three White Soldiers
        if (all(closes[i] > opens[i] and closes[i] > closes[i-1] for i in [-3, -2, -1]) and
            all((highs[i] - closes[i]) / (highs[i] - lows[i]) < 0.2 for i in [-3, -2, -1])):
            patterns.append('Three White Soldiers')
        # Three Black Crows
        if (all(closes[i] < opens[i] and closes[i] < closes[i-1] for i in [-3, -2, -1]) and
            all((closes[i] - lows[i]) / (highs[i] - lows[i]) < 0.2 for i in [-3, -2, -1])):
            patterns.append('Three Black Crows')

    return patterns

def detect_elliott_wave(highs, lows, closes):
    """Phát hiện mô hình Elliott Wave đơn giản"""
    wave_pattern = 'None'
    
    if len(closes) >= 10:
        # Tìm 5 sóng tăng (Wave 1-5)
        waves = []
        current_wave = 0
        wave_start = 0
        
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:  # Sóng tăng
                if current_wave == 0 or current_wave % 2 == 0:  # Bắt đầu sóng mới
                    current_wave += 1
                    wave_start = i-1
            elif closes[i] < closes[i-1]:  # Sóng giảm
                if current_wave % 2 == 1:  # Kết thúc sóng tăng
                    waves.append((wave_start, i-1, 'up'))
                    current_wave += 1
                    wave_start = i-1
        
        # Kiểm tra mô hình 5 sóng
        if len(waves) >= 5:
            # Kiểm tra quy tắc cơ bản của Elliott Wave
            wave1_length = waves[0][1] - waves[0][0]
            wave3_length = waves[2][1] - waves[2][0]
            wave5_length = waves[4][1] - waves[4][0]
            
            # Wave 3 thường là sóng dài nhất
            if wave3_length > wave1_length and wave3_length > wave5_length:
                wave_pattern = 'Elliott Wave 5 (Bullish)'
            else:
                wave_pattern = 'Elliott Wave 5 (Weak)'
    
    return wave_pattern

def analyze_timeframe(data, timeframe, current_price, symbol=None):
    """Phân tích kỹ thuật trên một khung thời gian với các chỉ báo và mô hình"""
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    open = data['open']

    # Chỉ báo động lượng - Giảm period để giảm lag
    rsi = talib.RSI(close, timeperiod=7)  # Từ 14 -> 7
    stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=7, slowk_period=3, slowd_period=3)  # Từ 14 -> 7
    macd, signal, _ = talib.MACD(close, fastperiod=6, slowperiod=13, signalperiod=4)  # Từ 12,26,9 -> 6,13,4
    cci = talib.CCI(high, low, close, timeperiod=7)  # Từ 14 -> 7
    roc = talib.ROC(close, timeperiod=6)  # Từ 12 -> 6

    # Chỉ báo xu hướng - Giảm period để phản ứng nhanh hơn
    sma20 = talib.SMA(close, timeperiod=20)  # Thêm SMA20
    sma50 = talib.SMA(close, timeperiod=50)
    ema20 = talib.EMA(close, timeperiod=20)  # Thêm EMA20
    ema50 = talib.EMA(close, timeperiod=50)
    wma20 = talib.WMA(close, timeperiod=20)  # Thêm WMA20
    wma50 = talib.WMA(close, timeperiod=50)
    ema100 = talib.EMA(close, timeperiod=100)  # Từ 200 -> 100
    adx = talib.ADX(high, low, close, timeperiod=7)  # Từ 14 -> 7
    sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
    upper, middle, lower = talib.BBANDS(close, timeperiod=10, nbdevup=2, nbdevdn=2)  # Từ 20 -> 10

    def calculate_ichimoku(high, low, close):
        tenkan = (talib.MAX(high, timeperiod=5) + talib.MIN(low, timeperiod=5)) / 2  # Từ 9 -> 5
        kijun = (talib.MAX(high, timeperiod=13) + talib.MIN(low, timeperiod=13)) / 2  # Từ 26 -> 13
        senkou_a = (tenkan + kijun) / 2
        senkou_b = (talib.MAX(high, timeperiod=26) + talib.MIN(low, timeperiod=26)) / 2  # Từ 52 -> 26
        chikou = close
        return tenkan, kijun, senkou_a, senkou_b, chikou

    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)

    # Chỉ báo khối lượng - Giảm period để phản ứng nhanh hơn
    obv = talib.OBV(close, volume)
    mfi = talib.MFI(high, low, close, volume, timeperiod=7)  # Từ 14 -> 7
    volume_profile = calculate_volume_profile(high, low, volume)
    vwap = calculate_vwap(high, low, close, volume)

    # Chỉ báo hỗn hợp - Giảm period để phản ứng nhanh hơn
    atr = talib.ATR(high, low, close, timeperiod=7)  # Từ 14 -> 7
    pivot_points = calculate_pivot_points(high, low, close)
    support, resistance = find_support_resistance(high, low, current_price)

    # Mô hình giá
    price_pattern = detect_price_patterns(high, low, close)

    # Mô hình nến Nhật
    candlestick_patterns = detect_candlestick_patterns(open, high, low, close)
    
    # Mô hình Elliott Wave
    elliott_wave = detect_elliott_wave(high, low, close)
    
    # Smart Money Concepts (SMC)
    order_blocks = detect_order_blocks(high, low, close, volume)
    fvgs = detect_fair_value_gaps(high, low, close)
    liquidity_zones = detect_liquidity_zones(high, low, close, volume)
    mitigation_zones = detect_mitigation_zones(high, low, close)
    
    # Price Action Patterns
    price_action_patterns = detect_price_action_patterns(high, low, close, volume)
    
    # Phân tích SMC và Price Action
    smc_signals = analyze_smc_signals(current_price, order_blocks, fvgs, liquidity_zones, mitigation_zones)
    pa_signals = analyze_price_action_signals(current_price, price_action_patterns, high, low, close)
    
    # Phân tích chi tiết mô hình nến
    candlestick_analysis = interpret_candlestick_patterns(candlestick_patterns, current_price, ema50[-1])
    
    # Tính toán điểm entry hợp lý
    entry_points = calculate_entry_points(current_price, high, low, close, rsi, upper, lower, ema50, pivot_points, support, resistance)

    candlestick_signal = 'Hold'
    if candlestick_analysis['bullish_count'] > candlestick_analysis['bearish_count']:
        candlestick_signal = 'Long'
    elif candlestick_analysis['bearish_count'] > candlestick_analysis['bullish_count']:
        candlestick_signal = 'Short'
    elif candlestick_analysis['neutral_count'] > 0:
        # Nếu chỉ có mô hình trung tính, phân tích chi tiết hơn
        for pattern in candlestick_patterns:
            if pattern in ['Doji', 'Spinning Top']:
                # Doji và Spinning Top cần xem xét vị trí
                if current_price > ema50[-1]:
                    candlestick_signal = 'Short'  # Ở vùng kháng cự
                else:
                    candlestick_signal = 'Long'   # Ở vùng hỗ trợ
                break

    # Tín hiệu từ các chỉ báo - Tối ưu để giảm lag và tăng độ nhạy
    rsi_signal = 'Hold'
    if rsi[-1] < 25:  # Từ 20 -> 25 (nhạy hơn)
        rsi_signal = 'Long'
    elif rsi[-1] > 75:  # Từ 80 -> 75 (nhạy hơn)
        rsi_signal = 'Short'

    stoch_signal = 'Hold'
    if stoch_k[-1] < 20 and stoch_k[-1] > stoch_d[-1]:  # Từ 15 -> 20
        stoch_signal = 'Long'
    elif stoch_k[-1] > 80 and stoch_k[-1] < stoch_d[-1]:  # Từ 85 -> 80
        stoch_signal = 'Short'

    macd_signal = 'Hold'
    if macd[-2] < signal[-2] and macd[-1] > signal[-1] and abs(macd[-1] - signal[-1]) > abs(macd[-2] - signal[-2]):
        macd_signal = 'Long'
    elif macd[-2] > signal[-2] and macd[-1] < signal[-1] and abs(macd[-1] - signal[-1]) > abs(macd[-2] - signal[-2]):
        macd_signal = 'Short'

    cci_signal = 'Hold'
    if cci[-1] < -100:  # Từ -150 -> -100 (nhạy hơn)
        cci_signal = 'Long'
    elif cci[-1] > 100:  # Từ 150 -> 100 (nhạy hơn)
        cci_signal = 'Short'

    roc_signal = 'Hold'
    if roc[-1] > 3:  # Từ 5 -> 3 (nhạy hơn)
        roc_signal = 'Long'
    elif roc[-1] < -3:  # Từ -5 -> -3 (nhạy hơn)
        roc_signal = 'Short'

    # Tín hiệu MA - Sử dụng MA ngắn hạn để giảm lag
    ma_signal = 'Hold'
    ma_distance = abs(sma20[-1] - ema100[-1]) / ema100[-1]
    if sma20[-1] > ema100[-1] and ema20[-1] > ema100[-1] and wma20[-1] > ema100[-1] and ma_distance > 0.01:
        ma_signal = 'Long'  # Sử dụng MA20 thay vì MA50
    elif sma20[-1] < ema100[-1] and ema20[-1] < ema100[-1] and wma20[-1] < ema100[-1] and ma_distance > 0.01:
        ma_signal = 'Short'  # Sử dụng MA20 thay vì MA50

    adx_signal = 'Hold'
    if adx[-1] > 25:  # Từ 35 -> 25 (nhạy hơn)
        if close[-1] > ema20[-1]:  # Sử dụng EMA20 thay vì EMA50
            adx_signal = 'Long'
        elif close[-1] < ema20[-1]:
            adx_signal = 'Short'

    # Thêm các chỉ báo leading (dẫn đầu) để giảm lag
    # 1. Williams %R - Chỉ báo momentum leading
    williams_r = talib.WILLR(high, low, close, timeperiod=7)
    williams_signal = 'Hold'
    if williams_r[-1] < -80:
        williams_signal = 'Long'
    elif williams_r[-1] > -20:
        williams_signal = 'Short'

    # 2. Ultimate Oscillator - Chỉ báo momentum leading
    ult_osc = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    ult_osc_signal = 'Hold'
    if ult_osc[-1] < 30:
        ult_osc_signal = 'Long'
    elif ult_osc[-1] > 70:
        ult_osc_signal = 'Short'

    # 3. Commodity Channel Index ngắn hạn
    cci_short = talib.CCI(high, low, close, timeperiod=5)
    cci_short_signal = 'Hold'
    if cci_short[-1] < -100:
        cci_short_signal = 'Long'
    elif cci_short[-1] > 100:
        cci_short_signal = 'Short'

    # 4. Momentum ngắn hạn
    momentum = talib.MOM(close, timeperiod=5)
    momentum_signal = 'Hold'
    if momentum[-1] > 0:
        momentum_signal = 'Long'
    elif momentum[-1] < 0:
        momentum_signal = 'Short'

    # 5. Chỉ báo đặc biệt cho hàng hóa (vàng và dầu)
    commodity_signals = {}
    if symbol in ['XAU/USD', 'WTI/USD']:
        # Aroon Indicator - tốt cho hàng hóa
        aroon_up, aroon_down = talib.AROON(high, low, timeperiod=14)
        aroon_signal = 'Hold'
        if aroon_up[-1] > 70 and aroon_down[-1] < 30:
            aroon_signal = 'Long'
        elif aroon_down[-1] > 70 and aroon_up[-1] < 30:
            aroon_signal = 'Short'
        commodity_signals['aroon_signal'] = aroon_signal
        
        # Commodity Selection Index (CSI) - chỉ báo đặc biệt cho hàng hóa
        # CSI = (ADX * ATR * 100) / (EMA * 100)
        adx_value = adx[-1] if not np.isnan(adx[-1]) else 25
        atr_value = atr[-1] if not np.isnan(atr[-1]) else np.mean(atr[-10:])
        ema_value = ema50[-1] if not np.isnan(ema50[-1]) else current_price
        
        if ema_value > 0:
            csi = (adx_value * atr_value * 100) / (ema_value * 100)
            csi_signal = 'Hold'
            if csi > 1000:  # CSI cao = xu hướng mạnh
                if close[-1] > ema50[-1]:
                    csi_signal = 'Long'
                else:
                    csi_signal = 'Short'
            commodity_signals['csi_signal'] = csi_signal
        else:
            commodity_signals['csi_signal'] = 'Hold'
        
        # Seasonal Analysis cho hàng hóa
        current_month = datetime.now().month
        seasonal_signal = 'Hold'
        
        if symbol == 'XAU/USD':  # Vàng
            # Vàng thường tăng vào tháng 1, 8, 9, 12
            bullish_months = [1, 8, 9, 12]
            bearish_months = [3, 4, 6, 7]
            if current_month in bullish_months:
                seasonal_signal = 'Long'
            elif current_month in bearish_months:
                seasonal_signal = 'Short'
        elif symbol == 'WTI/USD':  # Dầu
            # Dầu thường tăng vào mùa hè (6-8) và mùa đông (12-2)
            bullish_months = [1, 2, 6, 7, 8, 12]
            bearish_months = [3, 4, 5, 9, 10, 11]
            if current_month in bullish_months:
                seasonal_signal = 'Long'
            elif current_month in bearish_months:
                seasonal_signal = 'Short'
        
        commodity_signals['seasonal_signal'] = seasonal_signal

    sar_signal = 'Hold'
    sar_distance = abs(current_price - sar[-1]) / current_price
    if current_price > sar[-1] and sar_distance > 0.01:  # Thêm điều kiện khoảng cách
        sar_signal = 'Long'
    elif current_price < sar[-1] and sar_distance > 0.01:  # Thêm điều kiện khoảng cách
        sar_signal = 'Short'

    ichimoku_signal = 'Hold'
    try:
        if (len(senkou_a) > 0 and len(senkou_b) > 0 and len(tenkan) > 0 and len(kijun) > 0 and
            len(close) > 26):
            tenkan_kijun_distance = abs(tenkan[-1] - kijun[-1]) / kijun[-1]
            if (current_price > max(senkou_a[-1], senkou_b[-1]) and 
                tenkan[-1] > kijun[-1] and 
                close[-1] > close[-27] and tenkan_kijun_distance > 0.005):
                ichimoku_signal = 'Long'  # Thêm điều kiện khoảng cách Tenkan-Kijun
            elif (current_price < min(senkou_a[-1], senkou_b[-1]) and 
                  tenkan[-1] < kijun[-1] and 
                  close[-1] < close[-27] and tenkan_kijun_distance > 0.005):
                ichimoku_signal = 'Short'  # Thêm điều kiện khoảng cách Tenkan-Kijun
    except (IndexError, ValueError):
        ichimoku_signal = 'Hold'

    bb_signal = 'Hold'
    bb_width = (upper[-1] - lower[-1]) / middle[-1]
    if current_price <= lower[-1] * 0.995:  # Thêm điều kiện breakout mạnh hơn
        bb_signal = 'Long'
    elif current_price >= upper[-1] * 1.005:  # Thêm điều kiện breakout mạnh hơn
        bb_signal = 'Short'

    obv_signal = 'Hold'
    obv_slope = obv[-1] - obv[-10]
    obv_change = obv_slope / obv[-10] if obv[-10] != 0 else 0
    if obv_change > 0.05 and close[-1] > ema50[-1]:  # Thêm điều kiện thay đổi OBV
        obv_signal = 'Long'
    elif obv_change < -0.05 and close[-1] < ema50[-1]:  # Thêm điều kiện thay đổi OBV
        obv_signal = 'Short'

    mfi_signal = 'Hold'
    if mfi[-1] < 15:  # Từ 20 -> 15
        mfi_signal = 'Long'
    elif mfi[-1] > 85:  # Từ 80 -> 85
        mfi_signal = 'Short'

    volume_profile_signal = 'Hold'
    volume_distance = abs(current_price - volume_profile) / volume_profile
    if current_price > volume_profile and volume_distance > 0.02:  # Thêm điều kiện khoảng cách
        volume_profile_signal = 'Long'
    elif current_price < volume_profile and volume_distance > 0.02:  # Thêm điều kiện khoảng cách
        volume_profile_signal = 'Short'

    vwap_signal = 'Hold'
    vwap_distance = abs(current_price - vwap) / vwap
    if current_price > vwap and vwap_distance > 0.02:  # Thêm điều kiện khoảng cách
        vwap_signal = 'Long'
    elif current_price < vwap and vwap_distance > 0.02:  # Thêm điều kiện khoảng cách
        vwap_signal = 'Short'

    atr_signal = 'Hold'
    atr_avg = np.mean(atr[-10:])
    if atr[-1] > atr_avg * 1.5:  # Từ 1.2 -> 1.5 (biến động mạnh hơn)
        if close[-1] > ema50[-1]:
            atr_signal = 'Long'
        elif close[-1] < ema50[-1]:
            atr_signal = 'Short'

    pivot_signal = 'Hold'
    pivot_distance = min(abs(current_price - pivot_points['s1']), abs(current_price - pivot_points['r1'])) / current_price
    if current_price < pivot_points['s1'] and pivot_distance > 0.01:  # Thêm điều kiện khoảng cách
        pivot_signal = 'Long'
    elif current_price > pivot_points['r1'] and pivot_distance > 0.01:  # Thêm điều kiện khoảng cách
        pivot_signal = 'Short'

    wyckoff_signal = 'Hold'
    bb_width = (upper[-1] - lower[-1]) / middle[-1]
    if bb_width < 0.08 and current_price <= support * 1.01 and obv_change > 0.05:  # Từ 0.1 -> 0.08, thêm điều kiện OBV
        wyckoff_signal = 'Long'
    elif bb_width < 0.08 and current_price >= resistance * 0.99 and obv_change < -0.05:  # Từ 0.1 -> 0.08, thêm điều kiện OBV
        wyckoff_signal = 'Short'

    price_pattern_signal = 'Hold'
    if price_pattern in ['Head and Shoulders', 'Double Top']:
        price_pattern_signal = 'Short'
    elif price_pattern == 'Flag' and close[-1] > ema50[-1]:
        price_pattern_signal = 'Long'

    candlestick_signal = 'Hold'
    if any(p in ['Hammer', 'Bullish Engulfing', 'Morning Star', 'Three White Soldiers'] for p in candlestick_patterns):
        candlestick_signal = 'Long'
    elif any(p in ['Shooting Star', 'Bearish Engulfing', 'Evening Star', 'Three Black Crows'] for p in candlestick_patterns):
        candlestick_signal = 'Short'

    elliott_wave_signal = 'Hold'
    if 'Bullish' in elliott_wave:
        elliott_wave_signal = 'Long'
    elif 'Bearish' in elliott_wave:
        elliott_wave_signal = 'Short'

    signals = [
        rsi_signal, stoch_signal, macd_signal, cci_signal, roc_signal,
        ma_signal, adx_signal, sar_signal, ichimoku_signal, bb_signal,
        obv_signal, mfi_signal, volume_profile_signal, vwap_signal,
        atr_signal, pivot_signal, wyckoff_signal,
        price_pattern_signal, candlestick_signal, elliott_wave_signal,
        smc_signals['order_block_signal'], smc_signals['fvg_signal'], 
        smc_signals['liquidity_signal'], smc_signals['mitigation_signal'],
        pa_signals['pattern_signal'], pa_signals['momentum_signal'],
        # Thêm các chỉ báo leading mới
        williams_signal, ult_osc_signal, cci_short_signal, momentum_signal
    ]
    
    # Thêm các chỉ báo đặc biệt cho hàng hóa
    if symbol in ['XAU/USD', 'WTI/USD']:
        signals.extend([
            commodity_signals.get('aroon_signal', 'Hold'),
            commodity_signals.get('csi_signal', 'Hold'),
            commodity_signals.get('seasonal_signal', 'Hold')
        ])
    
    # Tăng trọng số cho các tín hiệu cực mạnh
    extra_signals = []
    
    # 1. RSI cực mạnh (quá mua/quá bán) - Nhạy hơn
    if rsi[-1] < 20:  # Từ 15 -> 20 (RSI cực thấp)
        extra_signals.extend(['Long', 'Long', 'Long'])  # Thêm 3 lần
    elif rsi[-1] > 80:  # Từ 85 -> 80 (RSI cực cao)
        extra_signals.extend(['Short', 'Short', 'Short'])  # Thêm 3 lần
    
    # 2. Stochastic cực mạnh - Nhạy hơn
    if stoch_k[-1] < 10:  # Từ 5 -> 10 (Stochastic cực thấp)
        extra_signals.extend(['Long', 'Long', 'Long'])
    elif stoch_k[-1] > 90:  # Từ 95 -> 90 (Stochastic cực cao)
        extra_signals.extend(['Short', 'Short', 'Short'])
    
    # 3. CCI cực mạnh - Nhạy hơn
    if cci[-1] < -150:  # Từ -250 -> -150 (CCI cực thấp)
        extra_signals.extend(['Long', 'Long', 'Long'])
    elif cci[-1] > 150:  # Từ 250 -> 150 (CCI cực cao)
        extra_signals.extend(['Short', 'Short', 'Short'])
    
    # 4. MFI cực mạnh - Nhạy hơn
    if mfi[-1] < 10:  # Từ 5 -> 10 (MFI cực thấp)
        extra_signals.extend(['Long', 'Long', 'Long'])
    elif mfi[-1] > 90:  # Từ 95 -> 90 (MFI cực cao)
        extra_signals.extend(['Short', 'Short', 'Short'])

    # 5. Williams %R cực mạnh (chỉ báo leading)
    if williams_r[-1] < -90:
        extra_signals.extend(['Long', 'Long', 'Long'])
    elif williams_r[-1] > -10:
        extra_signals.extend(['Short', 'Short', 'Short'])

    # 6. Ultimate Oscillator cực mạnh (chỉ báo leading)
    if ult_osc[-1] < 20:
        extra_signals.extend(['Long', 'Long', 'Long'])
    elif ult_osc[-1] > 80:
        extra_signals.extend(['Short', 'Short', 'Short'])

    # 7. Momentum cực mạnh (chỉ báo leading)
    if momentum[-1] > momentum[-2] * 1.5:  # Momentum tăng mạnh
        extra_signals.extend(['Long', 'Long', 'Long'])
    elif momentum[-1] < momentum[-2] * 0.5:  # Momentum giảm mạnh
        extra_signals.extend(['Short', 'Short', 'Short'])
    
    # 5. Bollinger Bands cực mạnh (breakout)
    bb_width = (upper[-1] - lower[-1]) / middle[-1]
    if current_price < lower[-1] * 0.985:  # Từ 0.99 -> 0.985 (Breakout xuống dưới BB mạnh hơn)
        extra_signals.extend(['Long', 'Long', 'Long'])
    elif current_price > upper[-1] * 1.015:  # Từ 1.01 -> 1.015 (Breakout lên trên BB mạnh hơn)
        extra_signals.extend(['Short', 'Short', 'Short'])
    
    # 6. Mô hình nến cực mạnh
    if candlestick_analysis['conclusion'].startswith('🟢') or candlestick_analysis['conclusion'].startswith('🔴'):
        # Thêm candlestick_signal 3 lần cho tín hiệu mạnh
        extra_signals.extend([candlestick_signal, candlestick_signal, candlestick_signal])
    
    # 7. ADX cực mạnh (xu hướng rất mạnh) - Nhạy hơn
    if adx[-1] > 30:  # Từ 50 -> 30 (Xu hướng cực mạnh)
        if close[-1] > ema20[-1]:  # Sử dụng EMA20 thay vì EMA50
            extra_signals.extend(['Long', 'Long', 'Long'])
        elif close[-1] < ema20[-1]:
            extra_signals.extend(['Short', 'Short', 'Short'])

    # 8. Price Action cực mạnh (breakout nhanh)
    if len(close) >= 3:
        price_change = (close[-1] - close[-3]) / close[-3]
        if price_change > 0.05:  # Tăng > 5% trong 3 nến
            extra_signals.extend(['Long', 'Long', 'Long'])
        elif price_change < -0.05:  # Giảm > 5% trong 3 nến
            extra_signals.extend(['Short', 'Short', 'Short'])
    
    # 8. MACD crossover cực mạnh
    if macd[-1] > signal[-1] * 1.2 and macd[-2] <= signal[-2]:  # Từ 1.1 -> 1.2 (Bullish crossover mạnh hơn)
        extra_signals.extend(['Long', 'Long', 'Long'])
    elif macd[-1] < signal[-1] * 0.8 and macd[-2] >= signal[-2]:  # Từ 0.9 -> 0.8 (Bearish crossover mạnh hơn)
        extra_signals.extend(['Short', 'Short', 'Short'])
    
    # 9. Volume breakout cực mạnh
    avg_volume = np.mean(volume[-20:])
    if volume[-1] > avg_volume * 5:  # Từ 3 -> 5 (Volume tăng 500%)
        if close[-1] > close[-2]:  # Giá tăng với volume lớn
            extra_signals.extend(['Long', 'Long', 'Long'])
        elif close[-1] < close[-2]:  # Giá giảm với volume lớn
            extra_signals.extend(['Short', 'Short', 'Short'])
    
    # 10. Pivot Points cực mạnh
    if current_price < pivot_points['s3']:  # Từ s2 -> s3 (Breakout dưới S3)
        extra_signals.extend(['Long', 'Long', 'Long'])
    elif current_price > pivot_points['r3']:  # Từ r2 -> r3 (Breakout trên R3)
        extra_signals.extend(['Short', 'Short', 'Short'])
    
    # 11. ROC cực mạnh (tốc độ thay đổi giá)
    if roc[-1] > 15:  # Từ 10 -> 15 (Tăng > 15%)
        extra_signals.extend(['Long', 'Long', 'Long'])
    elif roc[-1] < -15:  # Từ -10 -> -15 (Giảm > 15%)
        extra_signals.extend(['Short', 'Short', 'Short'])
    
    # 12. MA cực mạnh (khoảng cách lớn giữa các MA)
    ma_distance = abs(sma50[-1] - ema100[-1]) / ema100[-1]
    if ma_distance > 0.08:  # Từ 0.05 -> 0.08 (Khoảng cách > 8%)
        if sma50[-1] > ema100[-1]:
            extra_signals.extend(['Long', 'Long', 'Long'])
        else:
            extra_signals.extend(['Short', 'Short', 'Short'])
    
    # 13. SAR cực mạnh (khoảng cách lớn với giá)
    sar_distance = abs(current_price - sar[-1]) / current_price
    if sar_distance > 0.05:  # Từ 0.03 -> 0.05 (Khoảng cách > 5%)
        if current_price > sar[-1]:
            extra_signals.extend(['Long', 'Long', 'Long'])
        else:
            extra_signals.extend(['Short', 'Short', 'Short'])
    
    # 14. OBV cực mạnh (dòng tiền mạnh)
    obv_change = (obv[-1] - obv[-20]) / obv[-20]
    if obv_change > 0.15:  # Từ 0.1 -> 0.15 (OBV tăng > 15%)
        extra_signals.extend(['Long', 'Long', 'Long'])
    elif obv_change < -0.15:  # Từ -0.1 -> -0.15 (OBV giảm > 15%)
        extra_signals.extend(['Short', 'Short', 'Short'])
    
    # 15. Volume Profile cực mạnh (tập trung volume cao)
    volume_concentration = max(volume[-20:]) / np.mean(volume[-20:])
    if volume_concentration > 5:  # Từ 3 -> 5 (Volume tập trung > 500%)
        if current_price > volume_profile:
            extra_signals.extend(['Long', 'Long', 'Long'])
        else:
            extra_signals.extend(['Short', 'Short', 'Short'])
    
    # 16. VWAP cực mạnh (khoảng cách lớn với VWAP)
    vwap_distance = abs(current_price - vwap) / vwap
    if vwap_distance > 0.08:  # Từ 0.05 -> 0.08 (Khoảng cách > 8%)
        if current_price > vwap:
            extra_signals.extend(['Long', 'Long', 'Long'])
        else:
            extra_signals.extend(['Short', 'Short', 'Short'])
    
    # 17. ATR cực mạnh (biến động cực cao)
    atr_avg = np.mean(atr[-20:])
    if atr[-1] > atr_avg * 3:  # Từ 2 -> 3 (ATR > 300% trung bình)
        if close[-1] > ema50[-1]:
            extra_signals.extend(['Long', 'Long', 'Long'])
        else:
            extra_signals.extend(['Short', 'Short', 'Short'])
    
    # 18. Wyckoff cực mạnh (pattern tích lũy/phân phối rõ ràng)
    if bb_width < 0.05 and obv_change > 0.1:  # Từ 0.08 -> 0.05, từ 0.1 -> 0.1 (Tích lũy mạnh hơn)
        extra_signals.extend(['Long', 'Long', 'Long'])
    elif bb_width < 0.05 and obv_change < -0.1:  # Từ 0.08 -> 0.05, từ -0.1 -> -0.1 (Phân phối mạnh hơn)
        extra_signals.extend(['Short', 'Short', 'Short'])
    
    # 19. Price Pattern cực mạnh (mô hình giá rõ ràng)
    if price_pattern in ['Head and Shoulders', 'Double Top']:
        extra_signals.extend(['Short', 'Short', 'Short'])  # Thêm 3 lần cho mô hình đảo chiều mạnh
    elif price_pattern == 'Flag':
        extra_signals.extend(['Long', 'Long', 'Long'])  # Thêm 3 lần cho mô hình tiếp diễn
    
    # 20. Chỉ báo hàng hóa cực mạnh (cho vàng và dầu)
    if symbol in ['XAU/USD', 'WTI/USD']:
        # Aroon cực mạnh
        if commodity_signals.get('aroon_signal') == 'Long':
            extra_signals.extend(['Long', 'Long', 'Long'])
        elif commodity_signals.get('aroon_signal') == 'Short':
            extra_signals.extend(['Short', 'Short', 'Short'])
        
        # CSI cực mạnh
        if commodity_signals.get('csi_signal') == 'Long':
            extra_signals.extend(['Long', 'Long', 'Long'])
        elif commodity_signals.get('csi_signal') == 'Short':
            extra_signals.extend(['Short', 'Short', 'Short'])
        
        # Seasonal cực mạnh
        if commodity_signals.get('seasonal_signal') == 'Long':
            extra_signals.extend(['Long', 'Long', 'Long'])
        elif commodity_signals.get('seasonal_signal') == 'Short':
            extra_signals.extend(['Short', 'Short', 'Short'])

    # Thêm các tín hiệu cực mạnh vào danh sách
    signals.extend(extra_signals)
    
    long_count = signals.count('Long')
    short_count = signals.count('Short')
    total_signals = len(signals)

    signal = 'Hold'
    consensus_ratio = 0
    if long_count / total_signals >= SIGNAL_THRESHOLD:
        signal = 'Long'
        consensus_ratio = long_count / total_signals
    elif short_count / total_signals >= SIGNAL_THRESHOLD:
        signal = 'Short'
        consensus_ratio = short_count / total_signals

    return {
        'timeframe': timeframe,
        'signal': signal,
        'consensus_ratio': consensus_ratio,
        'rsi_signal': rsi_signal,
        'stoch_signal': stoch_signal,
        'macd_signal': macd_signal,
        'cci_signal': cci_signal,
        'roc_signal': roc_signal,
        'ma_signal': ma_signal,
        'adx_signal': adx_signal,
        'adx_value': adx[-1],
        'sar_signal': sar_signal,
        'ichimoku_signal': ichimoku_signal,
        'bb_signal': bb_signal,
        'obv_signal': obv_signal,
        'mfi_signal': mfi_signal,
        'volume_profile_signal': volume_profile_signal,
        'vwap_signal': vwap_signal,
        'atr_signal': atr_signal,
        'pivot_signal': pivot_signal,
        'wyckoff_signal': wyckoff_signal,
        'price_pattern_signal': price_pattern_signal,
        'candlestick_signal': candlestick_signal,
        'elliott_wave_signal': elliott_wave_signal,
        # Thêm các chỉ báo leading mới
        'williams_signal': williams_signal,
        'ult_osc_signal': ult_osc_signal,
        'cci_short_signal': cci_short_signal,
        'momentum_signal': momentum_signal,
        'rsi_value': rsi[-1],
        'mfi_value': mfi[-1],
        'current_price': current_price,
        'price_pattern': price_pattern,
        'candlestick_patterns': candlestick_patterns,
        'candlestick_analysis': candlestick_analysis,
        'elliott_wave': elliott_wave,
        'entry_points': entry_points,
        'smc_signals': smc_signals,
        'pa_signals': pa_signals,
        'order_blocks': order_blocks,
        'fvgs': fvgs,
        'liquidity_zones': liquidity_zones,
        'mitigation_zones': mitigation_zones,
        'price_action_patterns': price_action_patterns,
        'commodity_signals': commodity_signals if symbol in ['XAU/USD', 'WTI/USD'] else {}
    }

def make_decision(analyses):
    """Tổng hợp nhận định từ các khung thời gian
    
    Logic:
    - SIGNAL_THRESHOLD (50%): Ngưỡng tối thiểu để một timeframe được coi là có tín hiệu hợp lệ
    - consensus_ratio: Tỷ lệ đồng thuận thực tế của timeframe có tín hiệu mạnh nhất
    - Chỉ những timeframe có consensus_ratio >= SIGNAL_THRESHOLD mới được xét
    """
    valid_timeframes = []
    for analysis in analyses:
        if analysis['signal'] in ['Long', 'Short'] and analysis['consensus_ratio'] >= SIGNAL_THRESHOLD:
            valid_timeframes.append(analysis)
    
    if valid_timeframes:
        signals = [a['signal'] for a in valid_timeframes]
        has_long = 'Long' in signals
        has_short = 'Short' in signals
        if has_long and has_short:
            return 'Mixed', 0, valid_timeframes
        best_analysis = max(valid_timeframes, key=lambda x: x['consensus_ratio'])
        return best_analysis['signal'], best_analysis['consensus_ratio'], valid_timeframes
    return 'Hold', 0, []

def analyze_coin(symbol):
    """Phân tích xu hướng ngắn hạn cho một coin, vàng hoặc dầu"""
    try:
        logger.info(f"🔍 Bắt đầu phân tích {symbol}...")
        
        # Xử lý đặc biệt cho vàng và dầu
        if symbol in ['XAU/USD', 'WTI/USD']:
            current_price = get_commodity_current_price(symbol)
            if current_price is None:
                logger.error(f"Không thể lấy giá hiện tại cho {symbol}")
                return None
            logger.info(f"✅ Đã lấy giá hiện tại cho {symbol}: ${current_price}")
        else:
            # Xử lý bình thường cho crypto
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            logger.info(f"✅ Đã lấy giá hiện tại cho {symbol}: ${current_price}")
    except Exception as e:
        logger.error(f"Lỗi khi lấy giá hiện tại cho {symbol}: {e}")
        return None

    analyses = []
    for timeframe in TIMEFRAMES:
        logger.info(f"📊 Đang lấy dữ liệu {symbol} cho timeframe {timeframe}...")
        data = fetch_ohlcv(symbol, timeframe, CANDLE_LIMIT)
        if data is None:
            logger.warning(f"❌ Không thể lấy dữ liệu cho {symbol} ({timeframe})")
            continue
        logger.info(f"✅ Đã lấy dữ liệu {symbol} ({timeframe}): {len(data['close'])} candles")
        
        analysis = analyze_timeframe(data, timeframe, current_price, symbol)
        
        # Điều chỉnh phân tích dựa trên độ chính xác lịch sử
        analysis = adjust_analysis_based_on_accuracy(analysis, symbol, timeframe)
        
        analyses.append(analysis)

    if not analyses:
        logger.debug(f"Bỏ qua {symbol}: không có dữ liệu từ bất kỳ timeframe nào")
        return None

    decision, consensus_ratio, valid_timeframes = make_decision(analyses)

    # Tạo kết quả phân tích
    result = {
        'symbol': symbol,
        'decision': decision,
        'consensus_ratio': consensus_ratio,
        'valid_timeframes': valid_timeframes,
        'analyses': analyses,
        'current_price': current_price
    }

    # Lưu dự đoán cho các timeframe có tín hiệu mạnh
    for analysis in valid_timeframes:
        if analysis['consensus_ratio'] >= SIGNAL_THRESHOLD:
            prediction_data = {
                'trend': analysis.get('trend', 'neutral'),
                'signal': analysis.get('signal', 'Hold'),
                'confidence': analysis.get('confidence', 0),
                'entry_points': analysis.get('entry_points', {}),
                'analysis_summary': f"Timeframe: {analysis['timeframe']}, Consensus: {analysis['consensus_ratio']:.1%}",
                'technical_signals': {
                    'rsi_signal': analysis.get('rsi_signal', 'Hold'),
                    'macd_signal': analysis.get('macd_signal', 'Hold'),
                    'bb_signal': analysis.get('bb_signal', 'Hold'),
                    'stoch_signal': analysis.get('stoch_signal', 'Hold')
                },
                'commodity_signals': analysis.get('commodity_signals', {}),
                'smc_signals': analysis.get('smc_signals', {}),
                'price_action_signals': analysis.get('pa_signals', {})
            }
            
            # Lưu dự đoán
            prediction_id = save_prediction(symbol, analysis['timeframe'], prediction_data, current_price)
            if prediction_id:
                logger.info(f"📝 Đã lưu dự đoán {prediction_id} cho {symbol} ({analysis['timeframe']})")

    return result

def send_telegram_message(message):
    """Gửi tin nhắn qua Telegram Bot"""
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        logger.warning("Chưa cấu hình Telegram Bot Token hoặc Chat ID")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            logger.info("Đã gửi báo cáo qua Telegram thành công")
            return True
        else:
            logger.error(f"Lỗi khi gửi Telegram: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Lỗi khi gửi Telegram: {e}")
        return False

def format_coin_report(result):
    """Định dạng báo cáo phân tích cho một đồng coin, vàng hoặc dầu cụ thể - Tối ưu cho tín hiệu mạnh"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    symbol = result['symbol']
    decision = result['decision']
    consensus_ratio = result['consensus_ratio']
    valid_timeframes = result['valid_timeframes']
    
    # Xác định loại tài sản để hiển thị emoji phù hợp
    asset_type = "COIN"
    if symbol == 'XAU/USD':
        asset_type = "VÀNG"
    elif symbol == 'WTI/USD':
        asset_type = "DẦU"
    
    report = f"🤖 <b>PHÂN TÍCH {asset_type} {symbol}</b>\n"
    report += f"⏰ {current_time} | 📊 Ngưỡng tối thiểu: {SIGNAL_THRESHOLD:.1%}\n\n"
    
    if decision == 'Mixed':
        report += f"⚠️ <b>{symbol}: TÍN HIỆU TRÁI CHIỀU</b>\n"
        for analysis in valid_timeframes:
            report += f"  • {analysis['timeframe']}: {analysis['signal']} ({analysis['consensus_ratio']:.1%})\n"
    elif decision in ['Long', 'Short']:
        emoji = "✅" if decision == 'Long' else "🔴"
        report += f"{emoji} <b>{symbol}: {decision}</b> (Đồng thuận: {consensus_ratio:.1%})\n"
        report += f"📊 Timeframes: {', '.join([a['timeframe'] for a in valid_timeframes])}\n"
        report += f"💡 Tín hiệu từ timeframe có đồng thuận cao nhất\n\n"
        
        # Hiển thị thông tin về các timeframe được chọn
        for analysis in valid_timeframes:
            report += f"📊 <b>{analysis['timeframe']}:</b> {analysis['signal']} (Đồng thuận: {analysis['consensus_ratio']:.1%})\n"
        report += "\n"
        
        # Chỉ hiển thị các tín hiệu mạnh và quan trọng nhất
        for analysis in valid_timeframes:
            timeframe = analysis['timeframe']
            strong_signals = []
            
            # RSI cực mạnh (15/85)
            if analysis['rsi_value'] < 15 or analysis['rsi_value'] > 85:
                strong_signals.append(f"RSI({analysis['rsi_value']:.1f})")
            
            # Mô hình nến mạnh
            if analysis['candlestick_analysis']['conclusion'].startswith('🟢') or analysis['candlestick_analysis']['conclusion'].startswith('🔴'):
                strong_signals.append("Mô hình nến mạnh")
            
            # MACD crossover mạnh
            if analysis['macd_signal'] != 'Hold':
                strong_signals.append("MACD crossover")
            
            # Bollinger Bands breakout
            if analysis['bb_signal'] != 'Hold':
                strong_signals.append("BB breakout")
            
            # Mô hình giá quan trọng
            if analysis['price_pattern'] != 'None':
                strong_signals.append(f"Mô hình: {analysis['price_pattern']}")
            
            # SMC signals mạnh
            if 'smc_signals' in analysis:
                smc = analysis['smc_signals']
                if smc['order_block_signal'] != 'Hold':
                    strong_signals.append("Order Block")
                if smc['fvg_signal'] != 'Hold':
                    strong_signals.append("Fair Value Gap")
            
            # Price Action mạnh
            if 'pa_signals' in analysis:
                pa = analysis['pa_signals']
                if pa['pattern_signal'] != 'Hold':
                    strong_signals.append("Price Action")
            
            # Chỉ báo hàng hóa mạnh (cho vàng và dầu)
            if 'commodity_signals' in analysis and analysis['commodity_signals']:
                commodity = analysis['commodity_signals']
                if commodity.get('aroon_signal') != 'Hold':
                    strong_signals.append("Aroon")
                if commodity.get('csi_signal') != 'Hold':
                    strong_signals.append("CSI")
                if commodity.get('seasonal_signal') != 'Hold':
                    strong_signals.append("Seasonal")
            
            # Hiển thị tín hiệu mạnh (tối đa 5 tín hiệu quan trọng nhất)
            if strong_signals:
                report += f"📊 <b>{timeframe}:</b> {', '.join(strong_signals[:5])}\n"
            
            # Chỉ hiển thị entry points cho timeframe có tín hiệu mạnh nhất
            if analysis == max(valid_timeframes, key=lambda x: x['consensus_ratio']):
                if 'entry_points' in analysis:
                    entry = analysis['entry_points']
                    report += f"🎯 <b>ENTRY ({timeframe}):</b>\n"
                    report += f"  • Entry: ${entry['aggressive']:.4f}\n"
                    report += f"  • SL: ${entry['stop_loss']:.4f}\n"
                    report += f"  • TP: ${entry['take_profit']:.4f}\n"
            
            report += "\n"
    else:
        report += f"⏸️ {symbol}: Không có tín hiệu mạnh\n"
    
    return report

def format_analysis_report(results):
    """Định dạng báo cáo phân tích cho Telegram (giữ lại cho tương thích)"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Lấy thống kê độ chính xác
    accuracy_data = get_prediction_accuracy_data()
    accuracy_summary = ""
    if accuracy_data:
        overall = accuracy_data.get('overall', {})
        if overall.get('total_predictions', 0) > 0:
            accuracy_summary = f" | 📈 Độ chính xác: {overall['accuracy']:.1%} ({overall['accurate_predictions']}/{overall['total_predictions']})"
    
    report = f"🤖 <b>BÁO CÁO PHÂN TÍCH XU HƯỚNG</b>\n"
    report += f"⏰ Thời gian: {current_time}\n"
    report += f"📊 Ngưỡng tối thiểu: {SIGNAL_THRESHOLD:.1%}{accuracy_summary}\n"
    report += f"💰 Tài sản: Crypto, Vàng, Dầu\n\n"
    
    if not results:
        report += "📊 Không có xu hướng mạnh nào được phát hiện."
        return report
    
    for result in results:
        symbol = result['symbol']
        decision = result['decision']
        consensus_ratio = result['consensus_ratio']
        valid_timeframes = result['valid_timeframes']
        
        if decision == 'Mixed':
            report += f"⚠️ <b>{symbol}: CÓ KẾT LUẬN TRÁI CHIỀU</b>\n"
            for analysis in valid_timeframes:
                report += f"  • {analysis['timeframe']}: {analysis['signal']} ({analysis['consensus_ratio']:.1%})\n"
        elif decision in ['Long', 'Short']:
            emoji = "✅" if decision == 'Long' else "🔴"
            report += f"{emoji} <b>{symbol}: {decision}</b> (Đồng thuận: {consensus_ratio:.1%})\n"
            report += f"📊 Timeframes: {', '.join([a['timeframe'] for a in valid_timeframes])}\n"
            
            # Thêm thông tin chi tiết cho từng timeframe
            for analysis in valid_timeframes:
                timeframe = analysis['timeframe']
                report += f"📊 <b>Timeframe {timeframe}:</b>\n"
                report += f"📈 RSI: {analysis['rsi_signal']} ({analysis['rsi_value']:.1f}) -> {timeframe}\n"
                report += f"📊 MA: {analysis['ma_signal']} | ADX: {analysis['adx_signal']} -> {timeframe}\n"
                report += f"🎯 Ichimoku: {analysis['ichimoku_signal']} | SAR: {analysis['sar_signal']} -> {timeframe}\n"
                report += f"📉 BB: {analysis['bb_signal']} | OBV: {analysis['obv_signal']} -> {timeframe}\n"
                report += f"💰 MFI: {analysis['mfi_signal']} ({analysis['mfi_value']:.1f}) -> {timeframe}\n"
                
                if analysis['price_pattern'] != 'None':
                    report += f"📊 Mô hình giá: {analysis['price_pattern']} -> {timeframe}\n"
                if analysis['candlestick_patterns']:
                    report += f"🕯️ Mô hình nến: {', '.join(analysis['candlestick_patterns'])} -> {timeframe}\n"
                if analysis['candlestick_analysis']['conclusion'] != "⚪ KHÔNG CÓ MÔ HÌNH NẾN RÕ RÀNG":
                    report += f"📊 Phân tích mô hình nến: {analysis['candlestick_analysis']['conclusion']} -> {timeframe}\n"
                    report += f"📝 Chi tiết: {', '.join(analysis['candlestick_analysis']['analysis'])} -> {timeframe}\n"
                
                # Thêm thông tin chi tiết về các pattern khác
                if analysis['wyckoff_signal'] != 'Hold':
                    report += f"📈 Wyckoff: {analysis['wyckoff_signal']} -> {timeframe}\n"
                if analysis['pivot_signal'] != 'Hold':
                    report += f"🎯 Pivot: {analysis['pivot_signal']} -> {timeframe}\n"
                if analysis['elliott_wave'] != 'None':
                    report += f"🌊 Elliott Wave: {analysis['elliott_wave']} ({analysis['elliott_wave_signal']}) -> {timeframe}\n"
                
                # Chỉ báo hàng hóa (cho vàng và dầu)
                if 'commodity_signals' in analysis and analysis['commodity_signals']:
                    commodity = analysis['commodity_signals']
                    if commodity.get('aroon_signal') != 'Hold':
                        report += f"📈 Aroon: {commodity['aroon_signal']} -> {timeframe}\n"
                    if commodity.get('csi_signal') != 'Hold':
                        report += f"📊 CSI: {commodity['csi_signal']} -> {timeframe}\n"
                    if commodity.get('seasonal_signal') != 'Hold':
                        report += f"📅 Seasonal: {commodity['seasonal_signal']} -> {timeframe}\n"
                
                # Debug: Hiển thị tất cả các tín hiệu để kiểm tra
                report += f"🔍 Debug - Tất cả tín hiệu: RSI({analysis['rsi_signal']}), Stoch({analysis['stoch_signal']}), MACD({analysis['macd_signal']}), CCI({analysis['cci_signal']}), ROC({analysis['roc_signal']}), MA({analysis['ma_signal']}), ADX({analysis['adx_signal']}), SAR({analysis['sar_signal']}), Ichimoku({analysis['ichimoku_signal']}), BB({analysis['bb_signal']}), OBV({analysis['obv_signal']}), MFI({analysis['mfi_signal']}), VP({analysis['volume_profile_signal']}), VWAP({analysis['vwap_signal']}), ATR({analysis['atr_signal']}), Pivot({analysis['pivot_signal']}), Wyckoff({analysis['wyckoff_signal']}), Price({analysis['price_pattern_signal']}), Candle({analysis['candlestick_signal']}), Elliott({analysis['elliott_wave_signal']}) -> {timeframe}\n"
                
                # Thêm thông tin điểm entry
                if 'entry_points' in analysis:
                    entry = analysis['entry_points']
                    report += f"🎯 <b>ĐIỂM ENTRY HỢP LÝ ({timeframe}):</b>\n"
                    report += f"  • Entry bảo thủ: ${entry['conservative']:.4f}\n"
                    report += f"  • Entry tích cực: ${entry['aggressive']:.4f}\n"
                    report += f"  • Stop Loss: ${entry['stop_loss']:.4f}\n"
                    report += f"  • Take Profit: ${entry['take_profit']:.4f}\n"
                    for analysis_line in entry['analysis']:
                        report += f"  {analysis_line}\n"
                
                report += "\n"  # Thêm dòng trống giữa các timeframe
        else:
            report += f"⏸️ {symbol}: Không có tín hiệu mạnh\n"
        
        report += "\n"
    
    return report

def format_prediction_accuracy_report():
    """Định dạng báo cáo thống kê độ chính xác dự đoán"""
    accuracy_data = get_prediction_accuracy_data()
    if not accuracy_data:
        return "📊 Chưa có dữ liệu độ chính xác dự đoán"
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"📈 <b>BÁO CÁO ĐỘ CHÍNH XÁC DỰ ĐOÁN</b>\n"
    report += f"⏰ {current_time}\n\n"
    
    # Thống kê tổng thể
    overall = accuracy_data.get('overall', {})
    if overall.get('total_predictions', 0) > 0:
        report += f"📊 <b>THỐNG KÊ TỔNG THỂ:</b>\n"
        report += f"  • Tổng dự đoán: {overall['total_predictions']}\n"
        report += f"  • Dự đoán chính xác: {overall['accurate_predictions']}\n"
        report += f"  • Độ chính xác: {overall['accuracy']:.1%}\n\n"
    
    # Thống kê theo symbol
    symbol_stats = accuracy_data.get('by_symbol', {})
    if symbol_stats:
        report += f"💰 <b>THEO TÀI SẢN:</b>\n"
        for symbol, stats in symbol_stats.items():
            if stats['total'] > 0:
                emoji = "🟡" if symbol == 'XAU/USD' else "🟠" if symbol == 'WTI/USD' else "🟢"
                report += f"  {emoji} {symbol}: {stats['accuracy']:.1%} ({stats['accurate']}/{stats['total']})\n"
        report += "\n"
    
    # Thống kê theo timeframe
    timeframe_stats = accuracy_data.get('by_timeframe', {})
    if timeframe_stats:
        report += f"⏰ <b>THEO TIMEFRAME:</b>\n"
        for timeframe, stats in timeframe_stats.items():
            if stats['total'] > 0:
                report += f"  📊 {timeframe}: {stats['accuracy']:.1%} ({stats['accurate']}/{stats['total']})\n"
    
    return report

def telegram_report_scheduler():
    """Lập lịch gửi báo cáo Telegram định kỳ"""
    def send_periodic_report():
        while True:
            try:
                logger.info("🔄 Bắt đầu phân tích để gửi báo cáo Telegram...")
                
                results = []
                for symbol in SYMBOLS:
                    result = analyze_coin(symbol)
                    if result:
                        results.append(result)
                
                # Gửi báo cáo riêng cho từng coin
                if results:
                    for result in results:
                        try:
                            coin_report = format_coin_report(result)
                            send_telegram_message(coin_report)
                            logger.info(f"📱 Đã gửi báo cáo cho {result['symbol']}")
                            time.sleep(2)  # Chờ 2 giây giữa các tin nhắn
                        except Exception as e:
                            logger.error(f"Lỗi khi gửi báo cáo cho {result['symbol']}: {e}")
                else:
                    logger.info("Không có kết quả phân tích để gửi báo cáo")
                
                logger.info(f"⏰ Chờ {TELEGRAM_REPORT_INTERVAL} giây để gửi báo cáo tiếp theo...")
                time.sleep(TELEGRAM_REPORT_INTERVAL)
                
            except Exception as e:
                logger.error(f"Lỗi trong telegram_report_scheduler: {e}")
                time.sleep(60)  # Chờ 1 phút nếu có lỗi
    
    # Khởi động thread gửi báo cáo định kỳ
    report_thread = threading.Thread(target=send_periodic_report, daemon=True)
    report_thread.start()
    logger.info(f"📱 Đã khởi động Telegram Report Scheduler (gửi báo cáo mỗi {TELEGRAM_REPORT_INTERVAL//3600} giờ)")

def prediction_update_scheduler():
    """Lập lịch cập nhật kết quả dự đoán định kỳ"""
    def update_predictions_periodically():
        while True:
            try:
                logger.info("🔄 Cập nhật kết quả dự đoán...")
                
                # Cập nhật kết quả thực tế cho các dự đoán
                update_prediction_results()
                
                # Dọn dẹp các dự đoán cũ
                cleanup_old_predictions()
                
                logger.info(f"⏰ Chờ {PREDICTION_UPDATE_INTERVAL} giây để cập nhật dự đoán tiếp theo...")
                time.sleep(PREDICTION_UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Lỗi trong prediction_update_scheduler: {e}")
                time.sleep(300)  # Chờ 5 phút nếu có lỗi
    
    # Khởi động thread cập nhật dự đoán định kỳ
    prediction_thread = threading.Thread(target=update_predictions_periodically, daemon=True)
    prediction_thread.start()
    logger.info(f"📊 Đã khởi động Prediction Update Scheduler (cập nhật mỗi {PREDICTION_UPDATE_INTERVAL//3600} giờ)")

def calculate_entry_points(current_price, highs, lows, closes, rsi, bb_upper, bb_lower, ema50, pivot_points, support, resistance):
    """Tính toán các điểm entry hợp lý"""
    entry_points = {
        'immediate': current_price,
        'conservative': current_price,
        'aggressive': current_price,
        'stop_loss': current_price,
        'take_profit': current_price,
        'analysis': []
    }
    
    # 1. Phân tích xu hướng hiện tại
    trend = 'neutral'
    if current_price > ema50[-1]:
        trend = 'bullish'
    else:
        trend = 'bearish'
    
    # 2. Tính các mức entry cho Long
    if trend == 'bullish':
        # Entry bảo thủ (Conservative) - Chờ pullback về hỗ trợ
        conservative_entry = min(support, bb_lower[-1], pivot_points['s1'])
        entry_points['conservative'] = conservative_entry
        
        # Entry tích cực (Aggressive) - Vào ngay khi có tín hiệu
        aggressive_entry = current_price * 0.995  # Vào thấp hơn giá hiện tại 0.5%
        entry_points['aggressive'] = aggressive_entry
        
        # Stop Loss - Dựa trên mức hỗ trợ mạnh (s2) để tạo R/R tốt hơn
        # Sử dụng s2 thay vì s1 để SL gần entry hơn
        stop_loss = min(support * 0.998, bb_lower[-1] * 0.999, pivot_points['s2'] * 0.999)
        entry_points['stop_loss'] = stop_loss
        
        # Take Profit - Tỷ lệ với khoảng cách SL để tạo R/R ít nhất 1:2
        sl_distance = current_price - stop_loss
        if sl_distance > 0:
            # TP = Entry + (SL_distance * 2.5) để có R/R ít nhất 1:2.5
            take_profit = current_price + (sl_distance * 2.5)
        else:
            # Fallback nếu không tính được SL distance
            atr = np.mean([highs[i] - lows[i] for i in range(-10, 0)])
            take_profit = current_price + (atr * 1.5)
        entry_points['take_profit'] = take_profit
        
        entry_points['analysis'].append(f"📈 XU HƯỚNG TĂNG - Điểm entry hợp lý:")
        entry_points['analysis'].append(f"  • Entry bảo thủ: ${conservative_entry:.4f} (chờ pullback)")
        entry_points['analysis'].append(f"  • Entry tích cực: ${aggressive_entry:.4f} (vào ngay)")
        entry_points['analysis'].append(f"  • Stop Loss: ${stop_loss:.4f}")
        entry_points['analysis'].append(f"  • Take Profit: ${take_profit:.4f}")
    
    # 3. Tính các mức entry cho Short
    elif trend == 'bearish':
        # Entry bảo thủ - Chờ bounce về kháng cự
        conservative_entry = max(resistance, bb_upper[-1], pivot_points['r1'])
        entry_points['conservative'] = conservative_entry
        
        # Entry tích cực - Vào ngay khi có tín hiệu
        aggressive_entry = current_price * 1.005  # Vào cao hơn giá hiện tại 0.5%
        entry_points['aggressive'] = aggressive_entry
        
        # Stop Loss - Dựa trên mức kháng cự mạnh (r2) để tạo R/R tốt hơn
        # Sử dụng r2 thay vì r1 để SL gần entry hơn
        stop_loss = max(resistance * 1.002, bb_upper[-1] * 1.001, pivot_points['r2'] * 1.001)
        entry_points['stop_loss'] = stop_loss
        
        # Take Profit - Tỷ lệ với khoảng cách SL để tạo R/R ít nhất 1:2
        sl_distance = stop_loss - current_price
        if sl_distance > 0:
            # TP = Entry - (SL_distance * 2.5) để có R/R ít nhất 1:2.5
            take_profit = current_price - (sl_distance * 2.5)
        else:
            # Fallback nếu không tính được SL distance
            atr = np.mean([highs[i] - lows[i] for i in range(-10, 0)])
            take_profit = current_price - (atr * 1.5)
        entry_points['take_profit'] = take_profit
        
        entry_points['analysis'].append(f"📉 XU HƯỚNG GIẢM - Điểm entry hợp lý:")
        entry_points['analysis'].append(f"  • Entry bảo thủ: ${conservative_entry:.4f} (chờ bounce)")
        entry_points['analysis'].append(f"  • Entry tích cực: ${aggressive_entry:.4f} (vào ngay)")
        entry_points['analysis'].append(f"  • Stop Loss: ${stop_loss:.4f}")
        entry_points['analysis'].append(f"  • Take Profit: ${take_profit:.4f}")
    
    # 4. Phân tích RSI để tối ưu entry
    if rsi[-1] < 15:  # Từ 20 -> 15
        entry_points['analysis'].append(f"  • RSI quá bán ({rsi[-1]:.1f}) → Ưu tiên entry bảo thủ")
    elif rsi[-1] > 85:  # Từ 80 -> 85
        entry_points['analysis'].append(f"  • RSI quá mua ({rsi[-1]:.1f}) → Ưu tiên entry bảo thủ")
    else:
        entry_points['analysis'].append(f"  • RSI trung tính ({rsi[-1]:.1f}) → Có thể entry tích cực")
    
    # 5. Phân tích Bollinger Bands
    if current_price < bb_lower[-1]:
        entry_points['analysis'].append(f"  • Giá dưới BB Lower → Cơ hội entry tốt cho Long")
    elif current_price > bb_upper[-1]:
        entry_points['analysis'].append(f"  • Giá trên BB Upper → Cơ hội entry tốt cho Short")
    else:
        entry_points['analysis'].append(f"  • Giá trong BB → Entry ở giữa range")
    
    # 6. Tính Risk/Reward Ratio
    if trend == 'bullish':
        risk = current_price - entry_points['stop_loss']
        reward = entry_points['take_profit'] - current_price
        rr_ratio = reward / risk if risk > 0 else 0
        entry_points['analysis'].append(f"  • Risk/Reward Ratio: 1:{rr_ratio:.2f}")
    elif trend == 'bearish':
        risk = entry_points['stop_loss'] - current_price
        reward = current_price - entry_points['take_profit']
        rr_ratio = reward / risk if risk > 0 else 0
        entry_points['analysis'].append(f"  • Risk/Reward Ratio: 1:{rr_ratio:.2f}")
    
    return entry_points

def detect_order_blocks(highs, lows, closes, volumes):
    """Phát hiện Order Blocks (SMC)"""
    order_blocks = []
    
    for i in range(2, len(closes) - 1):
        # Bullish Order Block (sau khi giá tăng mạnh)
        if (closes[i+1] > closes[i] * 1.02 and  # Giá tăng > 2%
            volumes[i] > np.mean(volumes[max(0, i-10):i]) * 1.5):  # Volume cao
            
            # Tìm vùng order block (3-5 nến trước đó)
            block_start = max(0, i-5)
            block_high = max(highs[block_start:i+1])
            block_low = min(lows[block_start:i+1])
            
            order_blocks.append({
                'type': 'bullish',
                'start': block_start,
                'end': i,
                'high': block_high,
                'low': block_low,
                'strength': volumes[i] / np.mean(volumes[max(0, i-10):i])
            })
        
        # Bearish Order Block (sau khi giá giảm mạnh)
        elif (closes[i+1] < closes[i] * 0.98 and  # Giá giảm > 2%
              volumes[i] > np.mean(volumes[max(0, i-10):i]) * 1.5):  # Volume cao
            
            # Tìm vùng order block (3-5 nến trước đó)
            block_start = max(0, i-5)
            block_high = max(highs[block_start:i+1])
            block_low = min(lows[block_start:i+1])
            
            order_blocks.append({
                'type': 'bearish',
                'start': block_start,
                'end': i,
                'high': block_high,
                'low': block_low,
                'strength': volumes[i] / np.mean(volumes[max(0, i-10):i])
            })
    
    return order_blocks

def detect_fair_value_gaps(highs, lows, closes):
    """Phát hiện Fair Value Gaps (FVG) - SMC"""
    fvgs = []
    
    for i in range(1, len(closes) - 1):
        # Bullish FVG (gap lên)
        if lows[i+1] > highs[i-1]:
            fvgs.append({
                'type': 'bullish',
                'position': i,
                'gap_low': highs[i-1],
                'gap_high': lows[i+1],
                'size': lows[i+1] - highs[i-1],
                'filled': False
            })
        
        # Bearish FVG (gap xuống)
        elif highs[i+1] < lows[i-1]:
            fvgs.append({
                'type': 'bearish',
                'position': i,
                'gap_low': highs[i+1],
                'gap_high': lows[i-1],
                'size': lows[i-1] - highs[i+1],
                'filled': False
            })
    
    return fvgs

def detect_liquidity_zones(highs, lows, closes, volumes):
    """Phát hiện Liquidity Zones (SMC)"""
    liquidity_zones = []
    
    # Tìm các swing highs và lows
    for i in range(2, len(closes) - 2):
        # Swing High (liquidity trên)
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
            highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            
            # Kiểm tra volume và wick
            wick_size = (highs[i] - max(closes[i], closes[i])) / (highs[i] - lows[i])
            if wick_size > 0.3:  # Wick dài
                liquidity_zones.append({
                    'type': 'liquidity_high',
                    'position': i,
                    'price': highs[i],
                    'strength': wick_size,
                    'volume': volumes[i]
                })
        
        # Swing Low (liquidity dưới)
        elif (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
              lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            
            # Kiểm tra volume và wick
            wick_size = (min(closes[i], closes[i]) - lows[i]) / (highs[i] - lows[i])
            if wick_size > 0.3:  # Wick dài
                liquidity_zones.append({
                    'type': 'liquidity_low',
                    'position': i,
                    'price': lows[i],
                    'strength': wick_size,
                    'volume': volumes[i]
                })
    
    return liquidity_zones

def detect_mitigation_zones(highs, lows, closes):
    """Phát hiện Mitigation Zones (SMC) - vùng đảo chiều"""
    mitigation_zones = []
    
    for i in range(3, len(closes) - 3):
        # Bullish Mitigation (đảo chiều tăng)
        if (closes[i] > closes[i-1] and closes[i] > closes[i-2] and
            closes[i] > closes[i-3] and
            lows[i] < min(lows[i-3:i]) and  # Tạo đáy mới
            closes[i] > (highs[i-3] + lows[i-3]) / 2):  # Đóng trên midpoint
            
            mitigation_zones.append({
                'type': 'bullish_mitigation',
                'position': i,
                'price': closes[i],
                'strength': (closes[i] - lows[i]) / (highs[i] - lows[i])
            })
        
        # Bearish Mitigation (đảo chiều giảm)
        elif (closes[i] < closes[i-1] and closes[i] < closes[i-2] and
              closes[i] < closes[i-3] and
              highs[i] > max(highs[i-3:i]) and  # Tạo đỉnh mới
              closes[i] < (highs[i-3] + lows[i-3]) / 2):  # Đóng dưới midpoint
            
            mitigation_zones.append({
                'type': 'bearish_mitigation',
                'position': i,
                'price': closes[i],
                'strength': (highs[i] - closes[i]) / (highs[i] - lows[i])
            })
    
    return mitigation_zones

def detect_price_action_patterns(highs, lows, closes, volumes):
    """Phát hiện các mô hình Price Action"""
    patterns = []
    
    # Inside Bar
    for i in range(1, len(closes)):
        if (highs[i] <= highs[i-1] and lows[i] >= lows[i-1]):
            patterns.append({
                'type': 'inside_bar',
                'position': i,
                'strength': (highs[i-1] - lows[i-1]) / (highs[i] - lows[i]) if highs[i] != lows[i] else 1
            })
    
    # Outside Bar
    for i in range(1, len(closes)):
        if (highs[i] > highs[i-1] and lows[i] < lows[i-1]):
            patterns.append({
                'type': 'outside_bar',
                'position': i,
                'strength': (highs[i] - lows[i]) / (highs[i-1] - lows[i-1]) if highs[i-1] != lows[i-1] else 1
            })
    
    # Pin Bar (Hammer/Shooting Star)
    for i in range(1, len(closes)):
        body_size = abs(closes[i] - closes[i-1])
        total_range = highs[i] - lows[i]
        
        if total_range > 0:
            body_ratio = body_size / total_range
            upper_wick = highs[i] - max(closes[i], closes[i-1])
            lower_wick = min(closes[i], closes[i-1]) - lows[i]
            
            # Pin Bar (body nhỏ, wick dài)
            if body_ratio < 0.3:
                if upper_wick > body_size * 2 and lower_wick < body_size * 0.5:
                    patterns.append({
                        'type': 'pin_bar_bearish',
                        'position': i,
                        'strength': upper_wick / body_size
                    })
                elif lower_wick > body_size * 2 and upper_wick < body_size * 0.5:
                    patterns.append({
                        'type': 'pin_bar_bullish',
                        'position': i,
                        'strength': lower_wick / body_size
                    })
    
    # Engulfing Pattern
    for i in range(1, len(closes)):
        prev_body = abs(closes[i-1] - closes[i-2])
        curr_body = abs(closes[i] - closes[i-1])
        
        if curr_body > prev_body * 1.5:  # Body hiện tại lớn hơn 50%
            if closes[i] > closes[i-1] and closes[i-1] < closes[i-2]:  # Bullish Engulfing
                patterns.append({
                    'type': 'engulfing_bullish',
                    'position': i,
                    'strength': curr_body / prev_body
                })
            elif closes[i] < closes[i-1] and closes[i-1] > closes[i-2]:  # Bearish Engulfing
                patterns.append({
                    'type': 'engulfing_bearish',
                    'position': i,
                    'strength': curr_body / prev_body
                })
    
    return patterns

def analyze_smc_signals(current_price, order_blocks, fvgs, liquidity_zones, mitigation_zones):
    """Phân tích tín hiệu Smart Money Concepts"""
    smc_signals = {
        'order_block_signal': 'Hold',
        'fvg_signal': 'Hold',
        'liquidity_signal': 'Hold',
        'mitigation_signal': 'Hold',
        'analysis': []
    }
    
    # Phân tích Order Blocks
    nearby_bullish_ob = None
    nearby_bearish_ob = None
    
    for ob in order_blocks[-5:]:  # Chỉ xét 5 order blocks gần nhất
        if ob['type'] == 'bullish' and current_price >= ob['low'] and current_price <= ob['high']:
            nearby_bullish_ob = ob
        elif ob['type'] == 'bearish' and current_price >= ob['low'] and current_price <= ob['high']:
            nearby_bearish_ob = ob
    
    if nearby_bullish_ob:
        smc_signals['order_block_signal'] = 'Long'
        smc_signals['analysis'].append(f"📈 Bullish Order Block tại ${nearby_bullish_ob['low']:.4f} - ${nearby_bullish_ob['high']:.4f}")
    elif nearby_bearish_ob:
        smc_signals['order_block_signal'] = 'Short'
        smc_signals['analysis'].append(f"📉 Bearish Order Block tại ${nearby_bearish_ob['low']:.4f} - ${nearby_bearish_ob['high']:.4f}")
    
    # Phân tích Fair Value Gaps
    nearby_fvg = None
    for fvg in fvgs[-3:]:  # Chỉ xét 3 FVG gần nhất
        if fvg['gap_low'] <= current_price <= fvg['gap_high']:
            nearby_fvg = fvg
            break
    
    if nearby_fvg:
        if nearby_fvg['type'] == 'bullish':
            smc_signals['fvg_signal'] = 'Long'
            smc_signals['analysis'].append(f"📈 Bullish FVG tại ${nearby_fvg['gap_low']:.4f} - ${nearby_fvg['gap_high']:.4f}")
        else:
            smc_signals['fvg_signal'] = 'Short'
            smc_signals['analysis'].append(f"📉 Bearish FVG tại ${nearby_fvg['gap_low']:.4f} - ${nearby_fvg['gap_high']:.4f}")
    
    # Phân tích Liquidity Zones
    nearby_liquidity = None
    for lz in liquidity_zones[-3:]:  # Chỉ xét 3 liquidity zones gần nhất
        if abs(current_price - lz['price']) / lz['price'] < 0.02:  # Trong vòng 2%
            nearby_liquidity = lz
            break
    
    if nearby_liquidity:
        if nearby_liquidity['type'] == 'liquidity_high':
            smc_signals['liquidity_signal'] = 'Short'  # Có thể bị đảo chiều giảm
            smc_signals['analysis'].append(f"📉 Liquidity High tại ${nearby_liquidity['price']:.4f}")
        else:
            smc_signals['liquidity_signal'] = 'Long'  # Có thể bị đảo chiều tăng
            smc_signals['analysis'].append(f"📈 Liquidity Low tại ${nearby_liquidity['price']:.4f}")
    
    # Phân tích Mitigation Zones
    recent_mitigation = None
    for mz in mitigation_zones[-2:]:  # Chỉ xét 2 mitigation zones gần nhất
        if abs(current_price - mz['price']) / mz['price'] < 0.05:  # Trong vòng 5%
            recent_mitigation = mz
            break
    
    if recent_mitigation:
        if recent_mitigation['type'] == 'bullish_mitigation':
            smc_signals['mitigation_signal'] = 'Long'
            smc_signals['analysis'].append(f"📈 Bullish Mitigation tại ${recent_mitigation['price']:.4f}")
        else:
            smc_signals['mitigation_signal'] = 'Short'
            smc_signals['analysis'].append(f"📉 Bearish Mitigation tại ${recent_mitigation['price']:.4f}")
    
    return smc_signals

def analyze_price_action_signals(current_price, price_action_patterns, highs, lows, closes):
    """Phân tích tín hiệu Price Action"""
    pa_signals = {
        'pattern_signal': 'Hold',
        'momentum_signal': 'Hold',
        'analysis': []
    }
    
    # Phân tích các mô hình Price Action gần nhất
    recent_patterns = [p for p in price_action_patterns if p['position'] >= len(closes) - 5]
    
    bullish_patterns = 0
    bearish_patterns = 0
    
    for pattern in recent_patterns:
        if pattern['type'] in ['pin_bar_bullish', 'engulfing_bullish', 'outside_bar']:
            bullish_patterns += 1
        elif pattern['type'] in ['pin_bar_bearish', 'engulfing_bearish']:
            bearish_patterns += 1
    
    if bullish_patterns > bearish_patterns:
        pa_signals['pattern_signal'] = 'Long'
        pa_signals['analysis'].append(f"📈 {bullish_patterns} mô hình Price Action bullish")
    elif bearish_patterns > bullish_patterns:
        pa_signals['pattern_signal'] = 'Short'
        pa_signals['analysis'].append(f"📉 {bearish_patterns} mô hình Price Action bearish")
    
    # Phân tích momentum
    recent_closes = closes[-5:]
    if len(recent_closes) >= 3:
        momentum = (recent_closes[-1] - recent_closes[-3]) / recent_closes[-3]
        
        if momentum > 0.02:  # Tăng > 2%
            pa_signals['momentum_signal'] = 'Long'
            pa_signals['analysis'].append(f"📈 Momentum tăng {momentum:.2%}")
        elif momentum < -0.02:  # Giảm > 2%
            pa_signals['momentum_signal'] = 'Short'
            pa_signals['analysis'].append(f"📉 Momentum giảm {momentum:.2%}")
    
    return pa_signals

def main():
    logger.info("Bắt đầu phân tích xu hướng ngắn hạn trên Binance Spot...")
    
    # Khởi động Telegram Report Scheduler
    telegram_report_scheduler()
    
    # Khởi động Prediction Update Scheduler
    prediction_update_scheduler()
    
    symbols = get_usdt_symbols()
    logger.info(f"Đã chọn {len(symbols)} tài sản: {symbols}")
    logger.info("📊 Bao gồm: Crypto (BTC, ETH, BNB) từ Binance, Vàng & Dầu từ TradingView/Investing.com")

    # Phân tích lần đầu (chỉ để kiểm tra kết nối)
    results = []
    for symbol in symbols:
        result = analyze_coin(symbol)
        if result:
            results.append(result)
            # Hiển thị loại tài sản phù hợp
            if symbol == 'XAU/USD':
                logger.info(f"🟡 Đã phân tích Vàng {symbol} thành công")
            elif symbol == 'WTI/USD':
                logger.info(f"🟠 Đã phân tích Dầu {symbol} thành công")
            else:
                logger.info(f"✅ Đã phân tích {symbol} thành công")

    # Hiển thị thống kê độ chính xác nếu có
    accuracy_data = get_prediction_accuracy_data()
    if accuracy_data and accuracy_data.get('overall', {}).get('total_predictions', 0) > 0:
        overall = accuracy_data['overall']
        logger.info(f"📈 Thống kê độ chính xác: {overall['accuracy']:.1%} ({overall['accurate_predictions']}/{overall['total_predictions']})")
    
    logger.info(f"🤖 Bot đang chạy và gửi báo cáo Telegram mỗi {TELEGRAM_REPORT_INTERVAL//3600} giờ...")
    logger.info(f"📊 Hệ thống theo dõi dự đoán đang hoạt động (cập nhật mỗi {PREDICTION_UPDATE_INTERVAL//3600} giờ)")
    logger.info(f"📱 Nhấn Ctrl+C để dừng bot")
    
    # Giữ bot chạy để Telegram scheduler hoạt động
    try:
        while True:
            time.sleep(1800)  # Kiểm tra mỗi 30 phút
    except KeyboardInterrupt:
        logger.info(f"\n🛑 Bot đã dừng!")

if __name__ == "__main__":
    main()
