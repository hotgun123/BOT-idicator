import ccxt
import numpy as np
import pandas as pd
import requests
import threading
import yfinance as yf
from tradingview_ta import TA_Handler, Interval
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import time
import logging
import warnings
from dotenv import load_dotenv
import ta  # Thay thế TA-Lib với thư viện ta

# Suppress ML warnings
# The warning "No further splits with positive gain" is not an error - it means the model
# has reached its maximum potential with the current data and cannot find more useful splits.
# This is normal behavior and doesn't affect model performance.
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', message='.*No further splits with positive gain.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load environment variables
load_dotenv()

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo kết nối với Binance mainnet (spot) - Sửa lỗi 451
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
        'recvWindow': 60000,
    },
    'headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
})

# Thử sử dụng Binance Testnet nếu mainnet bị chặn
def get_exchange():
    """Lấy exchange với fallback options"""
    try:
        # Thử mainnet trước
        mainnet = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'recvWindow': 60000,
            },
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        })
        
        # Test kết nối
        mainnet.load_markets()
        logger.info("✅ Kết nối Binance mainnet thành công")
        return mainnet
        
    except Exception as e:
        logger.warning(f"⚠️ Không thể kết nối Binance mainnet: {e}")
        
        try:
            # Thử sử dụng yfinance làm fallback
            logger.info("🔄 Chuyển sang sử dụng yfinance...")
            return None  # Sẽ xử lý trong get_current_price
        except Exception as e2:
            logger.error(f"❌ Không thể kết nối bất kỳ exchange nào: {e2}")
            return None

# Khởi tạo exchange
exchange = get_exchange()

# Khởi tạo kết nối với Exness cho hàng hóa (đã loại bỏ)
exness_exchange = None

# Cấu hình
# Chỉ phân tích crypto; tạm thời bỏ vàng và dầu do nguồn dữ liệu không ổn định
SYMBOLS = ['BTC/USDT', 'ETH/USDT']  # Bỏ BNB theo yêu cầu của user
TIMEFRAMES = ['1h', '4h', '8h', '1d']  # Chỉ sử dụng 4 timeframe chính
ML_TIMEFRAMES = ['1h', '2h', '4h', '6h', '8h', '12h', '1d']  # Timeframes cho ML training
CANDLE_LIMIT = 1000
SIGNAL_THRESHOLD = 0.3 # Ngưỡng tối thiểu để một timeframe được coi là có tín hiệu hợp lệ
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

# === QUY CHUẨN TRADING SYSTEM - TIÊU CHUẨN MỚI ===
# Cấu hình cho hệ thống giao dịch đạt tiêu chuẩn
TRADING_SYSTEM_CONFIG = {
    'MIN_WIN_RATE': 0.6,  # Tỷ lệ thắng tối thiểu 60% (win probability > 1)
    'MIN_RR_RATIO': 1.0,  # Risk/Reward ratio tối thiểu 1:1
    'TARGET_RR_RATIO': 2.0,  # Mục tiêu RR ratio 1:2
    'MAX_RISK_PER_TRADE': 0.02,  # Rủi ro tối đa 2% mỗi lệnh
    'POSITION_SIZING_ENABLED': True,  # Bật tính toán position size
    'DYNAMIC_SL_TP': True,  # SL/TP động dựa trên volatility
    'TREND_CONFIRMATION_REQUIRED': True,  # Yêu cầu xác nhận xu hướng
    'MULTI_TIMEFRAME_CONFIRMATION': True,  # Xác nhận đa timeframe
    'VOLUME_CONFIRMATION_REQUIRED': True,  # Yêu cầu xác nhận volume
}

# Cấu hình cho validation và quality control
TRADE_QUALITY_CONFIG = {
    'MIN_SIGNAL_STRENGTH': 0.7,  # Độ mạnh tín hiệu tối thiểu
    'MIN_CONSENSUS_RATIO': 0.6,  # Tỷ lệ đồng thuận tối thiểu
    'MIN_INDICATOR_AGREEMENT': 0.6,  # Tỷ lệ đồng thuận chỉ báo tối thiểu
    'MAX_OPPOSING_SIGNALS': 2,  # Số tín hiệu ngược tối đa cho phép
    'REQUIRED_TIMEFRAMES': ['1h', '4h'],  # Timeframes bắt buộc phải đồng thuận
}

# Cấu hình cho risk management
RISK_MANAGEMENT_CONFIG = {
    'ATR_MULTIPLIER_SL': 1.5,  # Hệ số ATR cho Stop Loss
    'ATR_MULTIPLIER_TP': 3.0,  # Hệ số ATR cho Take Profit
    'SUPPORT_RESISTANCE_BUFFER': 0.005,  # Buffer 0.5% cho S/R levels
    'VOLATILITY_ADJUSTMENT': True,  # Điều chỉnh SL/TP theo volatility
    'BREAKOUT_CONFIRMATION': True,  # Xác nhận breakout trước khi vào lệnh
}

# === QUY CHUẨN TRADING SYSTEM - TIÊU CHUẨN MỚI ===
# Cấu hình cho hệ thống giao dịch đạt tiêu chuẩn
TRADING_SYSTEM_CONFIG = {
    'MIN_WIN_RATE': 0.6,  # Tỷ lệ thắng tối thiểu 60% (win probability > 1)
    'MIN_RR_RATIO': 1.0,  # Risk/Reward ratio tối thiểu 1:1
    'TARGET_RR_RATIO': 2.0,  # Mục tiêu RR ratio 1:2
    'MAX_RISK_PER_TRADE': 0.02,  # Rủi ro tối đa 2% mỗi lệnh
    'POSITION_SIZING_ENABLED': True,  # Bật tính toán position size
    'DYNAMIC_SL_TP': True,  # SL/TP động dựa trên volatility
    'TREND_CONFIRMATION_REQUIRED': True,  # Yêu cầu xác nhận xu hướng
    'MULTI_TIMEFRAME_CONFIRMATION': True,  # Xác nhận đa timeframe
    'VOLUME_CONFIRMATION_REQUIRED': True,  # Yêu cầu xác nhận volume
}

# Cấu hình cho validation và quality control
TRADE_QUALITY_CONFIG = {
    'MIN_SIGNAL_STRENGTH': 0.7,  # Độ mạnh tín hiệu tối thiểu
    'MIN_CONSENSUS_RATIO': 0.6,  # Tỷ lệ đồng thuận tối thiểu
    'MIN_INDICATOR_AGREEMENT': 0.6,  # Tỷ lệ đồng thuận chỉ báo tối thiểu
    'MAX_OPPOSING_SIGNALS': 2,  # Số tín hiệu ngược tối đa cho phép
    'REQUIRED_TIMEFRAMES': ['1h', '4h'],  # Timeframes bắt buộc phải đồng thuận
}

# Cấu hình cho risk management
RISK_MANAGEMENT_CONFIG = {
    'ATR_MULTIPLIER_SL': 1.5,  # Hệ số ATR cho Stop Loss
    'ATR_MULTIPLIER_TP': 3.0,  # Hệ số ATR cho Take Profit
    'SUPPORT_RESISTANCE_BUFFER': 0.005,  # Buffer 0.5% cho S/R levels
    'VOLATILITY_ADJUSTMENT': True,  # Điều chỉnh SL/TP theo volatility
    'BREAKOUT_CONFIRMATION': True,  # Xác nhận breakout trước khi vào lệnh
}

# Cấu hình Machine Learning
ML_MODELS_DIR = "ml_models"
ML_DATA_DIR = "ml_data"
ML_FEATURES_FILE = "ml_features.json"
ML_PERFORMANCE_FILE = "ml_performance.json"
ML_UPDATE_INTERVAL = 86400  # Cập nhật mô hình ML mỗi 24 giờ
ML_MIN_SAMPLES = 500  # Giảm xuống 500 để dễ train hơn
ML_CONFIDENCE_THRESHOLD = 0.7  # Ngưỡng tin cậy tối thiểu cho dự đoán ML
ML_HISTORICAL_CANDLES = 5000  # Số lượng candles lịch sử để train ML

# Cấu hình phân tích hội tụ (Convergence Analysis)
CONVERGENCE_ANALYSIS_ENABLED = True
CONVERGENCE_LOOKBACK_PERIODS = [5, 10, 20, 50]  # Các khoảng thời gian để phân tích hội tụ
CONVERGENCE_THRESHOLD = 0.8  # Ngưỡng hội tụ (0-1)
CONVERGENCE_WEIGHT = 0.3  # Trọng số cho tín hiệu hội tụ trong consensus

def get_usdt_symbols():
    """Trả về danh sách cặp giao dịch crypto"""
    return SYMBOLS

def ensure_prediction_data_dir():
    """Đảm bảo thư mục dữ liệu dự đoán tồn tại"""
    Path(PREDICTION_DATA_DIR).mkdir(exist_ok=True)

def ensure_ml_directories():
    """Đảm bảo các thư mục ML tồn tại"""
    Path(ML_MODELS_DIR).mkdir(exist_ok=True)
    Path(ML_DATA_DIR).mkdir(exist_ok=True)

def generate_trading_recommendation(current_price, support_price, resistance_price):
    """Tạo khuyến nghị giao dịch dựa trên vị trí giá so với support/resistance"""
    # Tính khoảng cách % đến các mức
    support_distance = ((current_price - support_price) / current_price) * 100
    resistance_distance = ((resistance_price - current_price) / current_price) * 100
    
    # Đưa ra khuyến nghị dựa trên vị trí giá
    if support_distance <= 2.0:  # Giá gần support (trong vòng 2%)
        return f"🟢 <b>LONG</b> - Giá gần vùng hỗ trợ mạnh ${support_price:.0f} (cách {support_distance:.1f}%)"
    elif resistance_distance <= 2.0:  # Giá gần resistance (trong vòng 2%)
        return f"🔴 <b>SHORT</b> - Giá gần vùng cản mạnh ${resistance_price:.0f} (cách {resistance_distance:.1f}%)"
    elif support_distance <= 5.0:  # Giá trong vùng support (trong vòng 5%)
        return f"🟡 <b>HOLD</b> - Giá trong vùng hỗ trợ ${support_price:.0f} (cách {support_distance:.1f}%)"
    elif resistance_distance <= 5.0:  # Giá trong vùng resistance (trong vòng 5%)
        return f"🟡 <b>HOLD</b> - Giá trong vùng cản ${resistance_price:.0f} (cách {resistance_distance:.1f}%)"
    else:  # Giá ở giữa
        return f"⚪ <b>NEUTRAL</b> - Giá ở giữa vùng giao dịch (S: ${support_price:.0f}, R: ${resistance_price:.0f})"

def create_ml_features(data, symbol, timeframe):
    """Tạo features cho Machine Learning từ dữ liệu OHLCV"""
    try:
        df = pd.DataFrame({
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'volume': data['volume']
        })
        
        # Technical Indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['ema_34'] = ta.trend.ema_indicator(df['close'], window=34)  # Elliott Wave main waves
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema_89'] = ta.trend.ema_indicator(df['close'], window=89)  # Elliott Wave main waves
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['volume_price_ratio'] = df['volume'] / df['close']
        
        # Moving averages
        df['ma_ratio_20_50'] = df['sma_20'] / df['sma_50']
        df['ema_ratio_20_50'] = df['ema_20'] / df['ema_50']
        
        # Bollinger Bands features
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # RSI features
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD features
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Momentum features
        df['momentum'] = df['close'] - df['close'].shift(5)
        df['rate_of_change'] = df['close'].pct_change(5)
        
        # Volatility features
        df['volatility'] = df['close'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['close']
        
        # Advanced features
        # Price patterns
        df['hammer'] = ((df['high'] - df['low']) > 3 * (df['open'] - df['close'])) & \
                      ((df['close'] - df['low']) / (0.001 + df['high'] - df['low']) > 0.6)
        df['doji'] = abs(df['open'] - df['close']) <= (df['high'] - df['low']) * 0.1
        
        # Support/Resistance levels
        df['support_level'] = df['low'].rolling(window=20).min()
        df['resistance_level'] = df['high'].rolling(window=20).max()
        df['support_distance'] = (df['close'] - df['support_level']) / df['close']
        df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        df['volume_z_score'] = (df['volume'] - df['volume_sma']) / (df['volume_std'] + 0.001)
        
        # Price momentum indicators
        df['price_acceleration'] = df['price_change'].diff()
        df['volume_price_trend'] = df['volume'] * df['price_change']
        
        # Market structure
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['trend_strength'] = df['higher_high'] - df['lower_low']
        
        # Target variable (next period's direction)
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        
        # Remove NaN values
        df = df.dropna()
        
        # Feature selection (35+ features)
        feature_columns = [
            # Technical indicators
            'rsi', 'macd', 'macd_signal', 'bb_position', 'bb_width',
            'ema_ratio_20_50', 'ma_ratio_20_50', 'stoch_k', 'stoch_d',
            'adx', 'atr', 'obv',
            
            # Price features
            'price_change', 'high_low_ratio', 'close_open_ratio',
            'rsi_oversold', 'rsi_overbought', 'macd_cross', 'macd_histogram',
            'momentum', 'rate_of_change', 'volatility_ratio',
            
            # Volume features
            'volume_ratio', 'volume_z_score', 'volume_price_trend',
            
            # Support/Resistance
            'support_distance', 'resistance_distance',
            
            # Market structure
            'trend_strength', 'price_acceleration',
            
            # Price patterns
            'hammer', 'doji'
        ]
        
        X = df[feature_columns]
        y = df['target']
        
        return X, y, feature_columns
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi tạo ML features cho {symbol}: {e}")
        return None, None, None

def train_ml_models(symbol, timeframe, force_full_update=False):
    """Train các mô hình Machine Learning với dữ liệu lịch sử (incremental update)"""
    try:
        ensure_ml_directories()
        
        # Kiểm tra độ mới của dữ liệu
        is_fresh, freshness_msg = check_data_freshness(symbol, timeframe)
        
        # Lấy dữ liệu lịch sử với incremental update
        data = load_and_update_historical_data(symbol, timeframe, force_full_update)
        if data is None:
            logger.warning(f"⚠️ Không thể lấy dữ liệu lịch sử cho {symbol} ({timeframe})")
            return None
            
        logger.info(f"📊 Dữ liệu {symbol} ({timeframe}): {len(data['close'])} candles ({freshness_msg})")
        
        if len(data['close']) < ML_MIN_SAMPLES:
            logger.warning(f"⚠️ Không đủ dữ liệu lịch sử để train ML cho {symbol} ({timeframe}): {len(data['close'])} < {ML_MIN_SAMPLES}")
            return None
        
        # Tạo features
        X, y, feature_columns = create_ml_features(data, symbol, timeframe)
        if X is None:
            logger.warning(f"⚠️ Không thể tạo features cho {symbol} ({timeframe})")
            return None
            
        logger.info(f"🔧 Features {symbol} ({timeframe}): {len(X)} samples, {len(feature_columns)} features")
        
        if len(X) < ML_MIN_SAMPLES:
            logger.warning(f"⚠️ Không đủ features để train ML cho {symbol} ({timeframe}): {len(X)} < {ML_MIN_SAMPLES}")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"📈 Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100, 
                random_state=42,
                verbose=-1,  # Suppress LightGBM output
                silent=True,  # Suppress LightGBM warnings
                min_child_samples=10,  # Minimum samples per leaf to avoid overfitting
                min_split_gain=0.0  # Allow splits with zero gain
            ),
            'logistic_regression': LogisticRegression(random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }
        
        trained_models = {}
        model_performance = {}
        
        # Train each model
        for name, model in models.items():
            try:
                logger.info(f"🔄 Training {name} cho {symbol} ({timeframe})...")
                
                # Suppress warnings during training
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    if name in ['svm', 'logistic_regression']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        y_proba = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate performance
                accuracy = accuracy_score(y_test, y_pred)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                model_performance[name] = {
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                trained_models[name] = {
                    'model': model,
                    'scaler': scaler if name in ['svm', 'logistic_regression'] else None,
                    'feature_columns': feature_columns,
                    'performance': model_performance[name]
                }
                
                logger.info(f"✅ {name} trained - Accuracy: {accuracy:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                
            except Exception as e:
                logger.error(f"❌ Lỗi khi train {name} cho {symbol}: {e}")
                continue
        
        # Save models
        safe_symbol = symbol.replace('/', '_')
        for name, model_data in trained_models.items():
            model_file = os.path.join(ML_MODELS_DIR, f"{safe_symbol}_{timeframe}_{name}.joblib")
            joblib.dump(model_data, model_file)
        
        # Save performance
        performance_file = os.path.join(ML_DATA_DIR, f"{safe_symbol}_{timeframe}_performance.json")
        with open(performance_file, 'w') as f:
            json.dump(model_performance, f, indent=2)
        
        logger.info(f"✅ Đã train và lưu {len(trained_models)} mô hình ML cho {symbol} ({timeframe})")
        return trained_models
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi train ML models cho {symbol}: {e}")
        return None

def predict_with_ml(symbol, timeframe, current_data):
    """Dự đoán sử dụng Machine Learning"""
    try:
        ensure_ml_directories()
        
        # Use safe symbol format (same as in train_ml_models)
        safe_symbol = symbol.replace('/', '_')
        
        # Load best performing model
        performance_file = os.path.join(ML_DATA_DIR, f"{safe_symbol}_{timeframe}_performance.json")
        if not os.path.exists(performance_file):
            logger.warning(f"⚠️ Không tìm thấy mô hình ML cho {symbol} ({timeframe})")
            return None
        
        with open(performance_file, 'r') as f:
            performance = json.load(f)
        
        # Find best model
        best_model_name = max(performance.keys(), key=lambda x: performance[x]['cv_mean'])
        best_model_file = os.path.join(ML_MODELS_DIR, f"{safe_symbol}_{timeframe}_{best_model_name}.joblib")
        
        if not os.path.exists(best_model_file):
            logger.warning(f"⚠️ Không tìm thấy file mô hình {best_model_name} cho {symbol}")
            return None
        
        # Load model
        model_data = joblib.load(best_model_file)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']
        
        # Create features for current data
        X_current, _, _ = create_ml_features(current_data, symbol, timeframe)
        if X_current is None or X_current.empty:
            return None
        
        # Get latest features
        latest_features = X_current[feature_columns].iloc[-1:].values
        
        # Scale if needed
        if scaler is not None:
            latest_features = scaler.transform(latest_features)
        
        # Make prediction
        prediction_proba = model.predict_proba(latest_features)[0]
        prediction_class = model.predict(latest_features)[0]
        
        # Calculate confidence
        confidence = max(prediction_proba)
        
        # Determine signal
        if prediction_class == 1 and confidence > ML_CONFIDENCE_THRESHOLD:
            signal = 'Long'
            predicted_direction = 'up'
        elif prediction_class == 0 and confidence > ML_CONFIDENCE_THRESHOLD:
            signal = 'Short'
            predicted_direction = 'down'
        else:
            signal = 'Hold'
            predicted_direction = 'sideways'
        
        # Tạo dữ liệu dự đoán để lưu với thông tin TP/SL
        current_price = current_data['close'][-1]
        prediction_data = {
            'predicted_price': current_price,
            'predicted_direction': predicted_direction,
            'features': feature_columns,
            'model_accuracy': performance[best_model_name]['cv_mean'],
            'prediction_horizon': timeframe,
            'current_price': current_price,  # Giá hiện tại làm entry price
            'target_profit_pct': 2.0,  # Mục tiêu lợi nhuận 2%
            'stop_loss_pct': 1.0,  # Cắt lỗ 1%
            'max_hold_time': '4h'  # Thời gian giữ lệnh tối đa
        }
        
        # Lưu dự đoán để đánh giá độ chính xác sau này
        prediction_id = save_ml_prediction(symbol, timeframe, prediction_data, confidence, best_model_name)
        
        # Điều chỉnh thuật toán dựa trên độ chính xác lịch sử
        adjusted_prediction = adjust_ml_algorithm_based_on_accuracy(symbol, timeframe, {
            'signal': signal,
            'confidence': confidence,
            'probability': prediction_proba,
            'model_name': best_model_name,
            'model_performance': performance[best_model_name],
            'predicted_direction': predicted_direction,
            'model_type': best_model_name
        })
        
        return adjusted_prediction
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi dự đoán ML cho {symbol}: {e}")
        return None

def analyze_convergence(data, lookback_periods=None):
    """Phân tích hội tụ (Convergence Analysis)"""
    if not CONVERGENCE_ANALYSIS_ENABLED:
        return None
    
    if lookback_periods is None:
        lookback_periods = CONVERGENCE_LOOKBACK_PERIODS
    
    try:
        # Convert to pandas Series if it's a numpy array
        close_prices = pd.Series(data['close']) if not isinstance(data['close'], pd.Series) else data['close']
        volume_data = pd.Series(data['volume']) if not isinstance(data['volume'], pd.Series) else data['volume']
        
        convergence_analysis = {
            'overall_convergence': 0.0,
            'period_convergence': {},
            'signals': [],
            'strength': 0.0
        }
        
        for period in lookback_periods:
            if len(close_prices) < period * 2:
                continue
            
            # Tính toán các chỉ số cho period này
            recent_prices = close_prices.iloc[-period:].values
            older_prices = close_prices.iloc[-period*2:-period].values
            
            # 1. Price Convergence
            recent_std = np.std(recent_prices)
            older_std = np.std(older_prices)
            price_convergence = 1 - (recent_std / older_std) if older_std > 0 else 0
            
            # 2. Volume Convergence
            recent_volume = volume_data.iloc[-period:].values
            older_volume = volume_data.iloc[-period*2:-period].values
            recent_vol_std = np.std(recent_volume)
            older_vol_std = np.std(older_volume)
            volume_convergence = 1 - (recent_vol_std / older_vol_std) if older_vol_std > 0 else 0
            
            # 3. Momentum Convergence
            recent_momentum = np.diff(recent_prices)
            older_momentum = np.diff(older_prices)
            # Remove NaN values
            recent_momentum = recent_momentum[~np.isnan(recent_momentum)]
            older_momentum = older_momentum[~np.isnan(older_momentum)]
            recent_mom_std = np.std(recent_momentum) if len(recent_momentum) > 0 else 0
            older_mom_std = np.std(older_momentum) if len(older_momentum) > 0 else 0
            momentum_convergence = 1 - (recent_mom_std / older_mom_std) if older_mom_std > 0 else 0
            
            # 4. RSI Convergence
            rsi = ta.momentum.rsi(close_prices, window=14)
            recent_rsi = rsi.iloc[-period:].values
            older_rsi = rsi.iloc[-period*2:-period].values
            # Remove NaN values
            recent_rsi = recent_rsi[~np.isnan(recent_rsi)]
            older_rsi = older_rsi[~np.isnan(older_rsi)]
            recent_rsi_std = np.std(recent_rsi) if len(recent_rsi) > 0 else 0
            older_rsi_std = np.std(older_rsi) if len(older_rsi) > 0 else 0
            rsi_convergence = 1 - (recent_rsi_std / older_rsi_std) if older_rsi_std > 0 else 0
            
            # 5. MACD Convergence
            macd = ta.trend.macd(close_prices)
            recent_macd = macd.iloc[-period:].values
            older_macd = macd.iloc[-period*2:-period].values
            # Remove NaN values
            recent_macd = recent_macd[~np.isnan(recent_macd)]
            older_macd = older_macd[~np.isnan(older_macd)]
            recent_macd_std = np.std(recent_macd) if len(recent_macd) > 0 else 0
            older_macd_std = np.std(older_macd) if len(older_macd) > 0 else 0
            macd_convergence = 1 - (recent_macd_std / older_macd_std) if older_macd_std > 0 else 0
            
            # Tính convergence tổng hợp cho period
            period_convergence = np.mean([
                price_convergence, volume_convergence, momentum_convergence,
                rsi_convergence, macd_convergence
            ])
            
            convergence_analysis['period_convergence'][f'{period}_periods'] = {
                'overall': period_convergence,
                'price': price_convergence,
                'volume': volume_convergence,
                'momentum': momentum_convergence,
                'rsi': rsi_convergence,
                'macd': macd_convergence
            }
            
            # Tạo tín hiệu dựa trên convergence
            if period_convergence > CONVERGENCE_THRESHOLD:
                # Convergence cao → có thể sắp breakout
                current_price = close_prices[-1]
                sma_20 = ta.trend.sma_indicator(close_prices, window=20).iloc[-1]
                
                if current_price > sma_20:
                    signal = 'Long'
                else:
                    signal = 'Short'
                
                convergence_analysis['signals'].append({
                    'period': period,
                    'signal': signal,
                    'strength': period_convergence,
                    'type': 'convergence_breakout'
                })
        
        # Tính convergence tổng thể
        if convergence_analysis['period_convergence']:
            overall_convergence = np.mean([
                data['overall'] for data in convergence_analysis['period_convergence'].values()
            ])
            convergence_analysis['overall_convergence'] = overall_convergence
            convergence_analysis['strength'] = overall_convergence
        
        return convergence_analysis
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi phân tích convergence: {e}")
        return None

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
        '1h': 3,    # Đánh giá sau 2 giờ
        '2h': 6,    # Đánh giá sau 4 giờ
        '4h': 12,    # Đánh giá sau 8 giờ
        '6h': 18,   # Đánh giá sau 12 giờ
        '8h': 24,   # Đánh giá sau 16 giờ
        '12h': 36,  # Đánh giá sau 24 giờ
        '1d': 72,   # Đánh giá sau 48 giờ
        '3d': 216,  # Đánh giá sau 6 ngày
        '1w': 672   # Đánh giá sau 14 ngày
    }
    return timeframe_hours.get(timeframe, 24) * 3600  # Chuyển sang giây

def get_current_price_for_prediction(symbol):
    """Lấy giá hiện tại cho việc cập nhật dự đoán với fallback"""
    try:
        if exchange:
            ticker = exchange.fetch_ticker(symbol)
            return ticker['last']
        else:
            # Fallback sử dụng yfinance
            symbol_mapping = {
                'BTC/USDT': 'BTC-USD',
                'ETH/USDT': 'ETH-USD'
            }
            
            yf_symbol = symbol_mapping.get(symbol, symbol.replace('/', '-'))
            ticker = yf.Ticker(yf_symbol)
            current_price = ticker.info.get('regularMarketPrice')
            
            if current_price:
                return current_price
            else:
                logger.error(f"❌ Không thể lấy giá từ yfinance cho {symbol}")
                return None
                
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy giá hiện tại cho {symbol}: {e}")
        
        # Thử fallback với yfinance
        try:
            symbol_mapping = {
                'BTC/USDT': 'BTC-USD',
                'ETH/USDT': 'ETH-USD'
            }
            
            yf_symbol = symbol_mapping.get(symbol, symbol.replace('/', '-'))
            ticker = yf.Ticker(yf_symbol)
            current_price = ticker.info.get('regularMarketPrice')
            
            if current_price:
                return current_price
        except Exception as e2:
            logger.error(f"❌ Fallback cũng thất bại cho {symbol}: {e2}")
        
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

def cleanup_old_data_files():
    """Giữ lại tất cả dữ liệu lịch sử để AI/ML học liên tục"""
    try:
        logger.info("🧹 Kiểm tra tình trạng dữ liệu lịch sử...")
        
        total_files = 0
        total_candles = 0
        total_size_mb = 0
        
        for symbol in ['BTC_USDT', 'ETH_USDT']:
            for timeframe in ML_TIMEFRAMES:
                data_file = os.path.join(ML_DATA_DIR, f"{symbol}_{timeframe}_historical.csv")
                
                if os.path.exists(data_file):
                    try:
                        df = pd.read_csv(data_file, index_col='timestamp', parse_dates=True)
                        file_size_mb = os.path.getsize(data_file) / (1024 * 1024)
                        
                        logger.info(f"📊 {symbol}_{timeframe}: {len(df)} candles, {file_size_mb:.2f}MB")
                        
                        total_files += 1
                        total_candles += len(df)
                        total_size_mb += file_size_mb
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Không thể đọc {data_file}: {e}")
        
        logger.info(f"📈 Tổng cộng: {total_files} files, {total_candles:,} candles, {total_size_mb:.2f}MB")
        logger.info("💡 Giữ lại tất cả dữ liệu lịch sử để AI/ML học liên tục")
            
    except Exception as e:
        logger.error(f"❌ Lỗi khi kiểm tra dữ liệu: {e}")

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

# Đã loại bỏ tất cả các hàm liên quan đến hàng hóa (vàng, dầu)

def fetch_ohlcv(symbol, timeframe, limit):
    """Lấy dữ liệu OHLCV cho crypto với fallback"""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            if exchange:
                # Xử lý cho crypto với Binance
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
            else:
                # Fallback sử dụng yfinance
                symbol_mapping = {
                    'BTC/USDT': 'BTC-USD',
                    'ETH/USDT': 'ETH-USD'
                }
                
                yf_symbol = symbol_mapping.get(symbol, symbol.replace('/', '-'))
                ticker = yf.Ticker(yf_symbol)
                
                # Chuyển đổi timeframe
                period_mapping = {
                    '1h': '1h',
                    '2h': '2h', 
                    '4h': '4h',
                    '6h': '6h',
                    '8h': '8h',
                    '12h': '12h',
                    '1d': '1d',
                    '3d': '3d',
                    '1w': '1wk'
                }
                
                period = period_mapping.get(timeframe, '1d')
                history = ticker.history(period=f"{limit}d", interval=period)
                
                if len(history) < limit * 0.5:
                    logger.warning(f"⚠️ Dữ liệu yfinance cho {symbol} ({timeframe}) không đủ: {len(history)}/{limit}")
                    return None
                
                return {
                    'open': history['Open'].values,
                    'high': history['High'].values,
                    'low': history['Low'].values,
                    'close': history['Close'].values,
                    'volume': history['Volume'].values
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
                
                # Thử fallback với yfinance
                try:
                    symbol_mapping = {
                        'BTC/USDT': 'BTC-USD',
                        'ETH/USDT': 'ETH-USD'
                    }
                    
                    yf_symbol = symbol_mapping.get(symbol, symbol.replace('/', '-'))
                    ticker = yf.Ticker(yf_symbol)
                    period = period_mapping.get(timeframe, '1d')
                    history = ticker.history(period=f"{limit}d", interval=period)
                    
                    if len(history) > 0:
                        return {
                            'open': history['Open'].values,
                            'high': history['High'].values,
                            'low': history['Low'].values,
                            'close': history['Close'].values,
                            'volume': history['Volume'].values
                        }
                except Exception as e2:
                    logger.error(f"❌ Fallback yfinance cũng thất bại cho {symbol}: {e2}")
    
    return None

def calculate_fibonacci_levels(highs, lows):
    """Tính các mức Fibonacci Retracement theo phương pháp trading Việt Nam"""
    try:
        max_price = max(highs[-50:])
        min_price = min(lows[-50:])
        diff = max_price - min_price
        
        # Fibonacci Retracement levels theo chuẩn trading Việt Nam
        levels = {
            '0%': max_price,
            '23.6%': max_price - 0.236 * diff,
            '38.2%': max_price - 0.382 * diff,
            '50%': max_price - 0.5 * diff,
            '61.8%': max_price - 0.618 * diff,
            '76.4%': max_price - 0.764 * diff,
            '100%': min_price
        }
        return levels
    except Exception as e:
        logger.error(f"Lỗi tính Fibonacci levels: {e}")
        return {}

def calculate_fibonacci_extension(highs, lows):
    """Tính các mức Fibonacci Extension cho target prediction"""
    try:
        max_price = max(highs[-50:])
        min_price = min(lows[-50:])
        diff = max_price - min_price
        
        # Fibonacci Extension levels (dùng khi xu thế cấp 1 tiếp diễn)
        extension_levels = {
            '61.8%': max_price + 0.618 * diff,
            '100%': max_price + diff,
            '161.8%': max_price + 1.618 * diff,
            '261.8%': max_price + 2.618 * diff,
            '361.8%': max_price + 3.618 * diff,
            '461.8%': max_price + 4.618 * diff
        }
        return extension_levels
    except Exception as e:
        logger.error(f"Lỗi tính Fibonacci Extension: {e}")
        return {}

def analyze_fibonacci_psychology(current_price, fib_levels, price_range):
    """Phân tích tâm lý thị trường dựa trên Fibonacci"""
    try:
        psychology_analysis = {
            'market_sentiment': 'neutral',
            'buyer_strength': 0.5,
            'seller_strength': 0.5,
            'key_level': None,
            'analysis': ''
        }
        
        # Xác định vùng Fibonacci hiện tại
        if current_price >= fib_levels.get('0%', 0):
            psychology_analysis['market_sentiment'] = 'strong_bullish'
            psychology_analysis['buyer_strength'] = 1.0
            psychology_analysis['analysis'] = 'Giá vượt đỉnh - phe mua rất mạnh'
        elif current_price <= fib_levels.get('100%', 0):
            psychology_analysis['market_sentiment'] = 'strong_bearish'
            psychology_analysis['seller_strength'] = 1.0
            psychology_analysis['analysis'] = 'Giá chạm đáy - phe bán rất mạnh'
        elif current_price <= fib_levels.get('23.6%', 0):
            psychology_analysis['market_sentiment'] = 'bullish'
            psychology_analysis['buyer_strength'] = 0.8
            psychology_analysis['key_level'] = '23.6%'
            psychology_analysis['analysis'] = 'Hồi về 23.6% - phe mua vẫn mạnh'
        elif current_price <= fib_levels.get('38.2%', 0):
            psychology_analysis['market_sentiment'] = 'slightly_bullish'
            psychology_analysis['buyer_strength'] = 0.6
            psychology_analysis['key_level'] = '38.2%'
            psychology_analysis['analysis'] = 'Hồi về 38.2% - phe mua hơi yếu'
        elif current_price <= fib_levels.get('50%', 0):
            psychology_analysis['market_sentiment'] = 'neutral'
            psychology_analysis['buyer_strength'] = 0.5
            psychology_analysis['key_level'] = '50%'
            psychology_analysis['analysis'] = 'Hồi về 50% - cân bằng mua bán'
        elif current_price <= fib_levels.get('61.8%', 0):
            psychology_analysis['market_sentiment'] = 'slightly_bearish'
            psychology_analysis['buyer_strength'] = 0.4
            psychology_analysis['key_level'] = '61.8%'
            psychology_analysis['analysis'] = 'Hồi về 61.8% - phe mua yếu sinh lý'
        elif current_price <= fib_levels.get('76.4%', 0):
            psychology_analysis['market_sentiment'] = 'bearish'
            psychology_analysis['buyer_strength'] = 0.2
            psychology_analysis['key_level'] = '76.4%'
            psychology_analysis['analysis'] = 'Hồi về 76.4% - phe mua rất yếu'
        
        return psychology_analysis
    except Exception as e:
        logger.error(f"Lỗi phân tích tâm lý Fibonacci: {e}")
        return {'market_sentiment': 'neutral', 'buyer_strength': 0.5, 'seller_strength': 0.5, 'key_level': None, 'analysis': ''}

def analyze_ema_34_89_trend(close, ema34, ema89, current_price):
    """Phân tích xu hướng dựa trên EMA 34 và EMA 89 theo Elliott Wave theory"""
    try:
        trend_analysis = {
            'trend_direction': 'neutral',
            'trend_strength': 0.5,
            'value_zone': False,
            'entry_signal': 'hold',
            'analysis': ''
        }
        
        current_ema34 = ema34.iloc[-1] if hasattr(ema34, 'iloc') else ema34[-1]
        current_ema89 = ema89.iloc[-1] if hasattr(ema89, 'iloc') else ema89[-1]
        
        # Xác định xu hướng chính
        if current_price > current_ema34 and current_ema34 > current_ema89:
            trend_analysis['trend_direction'] = 'bullish'
            trend_analysis['trend_strength'] = 0.8
            trend_analysis['entry_signal'] = 'buy'
            trend_analysis['analysis'] = 'Xu hướng tăng mạnh - giá trên EMA34, EMA34 trên EMA89'
        elif current_price < current_ema34 and current_ema34 < current_ema89:
            trend_analysis['trend_direction'] = 'bearish'
            trend_analysis['trend_strength'] = 0.8
            trend_analysis['entry_signal'] = 'sell'
            trend_analysis['analysis'] = 'Xu hướng giảm mạnh - giá dưới EMA34, EMA34 dưới EMA89'
        elif current_price > current_ema34 and current_ema34 < current_ema89:
            trend_analysis['trend_direction'] = 'mixed'
            trend_analysis['trend_strength'] = 0.4
            trend_analysis['entry_signal'] = 'hold'
            trend_analysis['analysis'] = 'Tín hiệu hỗn hợp - cần xác nhận thêm'
        elif current_price < current_ema34 and current_ema34 > current_ema89:
            trend_analysis['trend_direction'] = 'mixed'
            trend_analysis['trend_strength'] = 0.4
            trend_analysis['entry_signal'] = 'hold'
            trend_analysis['analysis'] = 'Tín hiệu hỗn hợp - cần xác nhận thêm'
        
        # Phân tích vùng giá trị (value zone)
        ema_distance = abs(current_price - current_ema34) / current_ema34
        if ema_distance > 0.02:  # Giá cách EMA34 hơn 2%
            trend_analysis['value_zone'] = True
            trend_analysis['analysis'] += ' - Vùng giá trị: giá xa EMA34'
        
        return trend_analysis
    except Exception as e:
        logger.error(f"Lỗi phân tích EMA 34/89: {e}")
        return {'trend_direction': 'neutral', 'trend_strength': 0.5, 'value_zone': False, 'entry_signal': 'hold', 'analysis': ''}

def detect_ema_breakout_pattern(close, ema34, ema89):
    """Phát hiện mô hình breakout EMA theo phương pháp trading Việt Nam"""
    try:
        breakout_analysis = {
            'pattern': 'none',
            'signal': 'hold',
            'strength': 0.0,
            'analysis': ''
        }
        
        if len(close) < 5:
            return breakout_analysis
        
        # Lấy 5 giá trị gần nhất
        recent_closes = close.iloc[-5:] if hasattr(close, 'iloc') else close[-5:]
        recent_ema34 = ema34.iloc[-5:] if hasattr(ema34, 'iloc') else ema34[-5:]
        recent_ema89 = ema89.iloc[-5:] if hasattr(ema89, 'iloc') else ema89[-5:]
        
        current_close = recent_closes.iloc[-1] if hasattr(recent_closes, 'iloc') else recent_closes[-1]
        current_ema34 = recent_ema34.iloc[-1] if hasattr(recent_ema34, 'iloc') else recent_ema34[-1]
        current_ema89 = recent_ema89.iloc[-1] if hasattr(recent_ema89, 'iloc') else recent_ema89[-1]
        
        # Kiểm tra mô hình breakout tăng
        if (recent_closes.iloc[-2] if hasattr(recent_closes, 'iloc') else recent_closes[-2]) < (recent_ema34.iloc[-2] if hasattr(recent_ema34, 'iloc') else recent_ema34[-2]) and \
           current_close > current_ema34:
            breakout_analysis['pattern'] = 'bullish_breakout'
            breakout_analysis['signal'] = 'buy'
            breakout_analysis['strength'] = 0.8
            breakout_analysis['analysis'] = 'Breakout tăng qua EMA34 - tín hiệu mua mạnh'
        
        # Kiểm tra mô hình breakout giảm
        elif (recent_closes.iloc[-2] if hasattr(recent_closes, 'iloc') else recent_closes[-2]) > (recent_ema34.iloc[-2] if hasattr(recent_ema34, 'iloc') else recent_ema34[-2]) and \
             current_close < current_ema34:
            breakout_analysis['pattern'] = 'bearish_breakout'
            breakout_analysis['signal'] = 'sell'
            breakout_analysis['strength'] = 0.8
            breakout_analysis['analysis'] = 'Breakout giảm qua EMA34 - tín hiệu bán mạnh'
        
        # Kiểm tra mô hình pullback (vòng về EMA)
        elif abs(current_close - current_ema34) / current_ema34 < 0.005:  # Giá gần EMA34
            breakout_analysis['pattern'] = 'pullback'
            breakout_analysis['signal'] = 'hold'
            breakout_analysis['strength'] = 0.6
            breakout_analysis['analysis'] = 'Giá vòng về EMA34 - chờ xác nhận'
        
        return breakout_analysis
    except Exception as e:
        logger.error(f"Lỗi phát hiện mô hình breakout EMA: {e}")
        return {'pattern': 'none', 'signal': 'hold', 'strength': 0.0, 'analysis': ''}

def analyze_ma_value_zones(close, ema34, ema89, current_price):
    """Phân tích vùng giá trị và hành vi giá theo phương pháp trading Việt Nam"""
    try:
        value_zone_analysis = {
            'zone_type': 'neutral',
            'distance_from_ma': 0.0,
            'price_behavior': 'normal',
            'entry_opportunity': 'none',
            'analysis': ''
        }
        
        current_ema34 = ema34.iloc[-1] if hasattr(ema34, 'iloc') else ema34[-1]
        current_ema89 = ema89.iloc[-1] if hasattr(ema89, 'iloc') else ema89[-1]
        
        # Tính khoảng cách từ giá đến EMA34
        distance_percent = abs(current_price - current_ema34) / current_ema34
        value_zone_analysis['distance_from_ma'] = distance_percent
        
        # Xác định loại vùng giá trị
        if distance_percent < 0.005:  # Giá gần EMA34 (< 0.5%)
            value_zone_analysis['zone_type'] = 'value_zone'
            value_zone_analysis['entry_opportunity'] = 'high'
            value_zone_analysis['analysis'] = 'Vùng giá trị - cơ hội entry tốt'
        elif distance_percent < 0.02:  # Giá gần EMA34 (< 2%)
            value_zone_analysis['zone_type'] = 'near_value'
            value_zone_analysis['entry_opportunity'] = 'medium'
            value_zone_analysis['analysis'] = 'Gần vùng giá trị - cơ hội entry trung bình'
        elif distance_percent > 0.05:  # Giá xa EMA34 (> 5%)
            value_zone_analysis['zone_type'] = 'extreme'
            value_zone_analysis['entry_opportunity'] = 'low'
            value_zone_analysis['analysis'] = 'Giá xa EMA34 - cơ hội entry thấp'
        
        # Phân tích hành vi giá
        if len(close) >= 3:
            recent_closes = close.iloc[-3:] if hasattr(close, 'iloc') else close[-3:]
            recent_ema34 = ema34.iloc[-3:] if hasattr(ema34, 'iloc') else ema34[-3:]
            
            # Kiểm tra giá có vòng về EMA không
            if (recent_closes.iloc[-2] if hasattr(recent_closes, 'iloc') else recent_closes[-2]) < (recent_ema34.iloc[-2] if hasattr(recent_ema34, 'iloc') else recent_ema34[-2]) and \
               current_price > current_ema34:
                value_zone_analysis['price_behavior'] = 'pullback_bullish'
                value_zone_analysis['analysis'] += ' - Giá vòng lên qua EMA34'
            elif (recent_closes.iloc[-2] if hasattr(recent_closes, 'iloc') else recent_closes[-2]) > (recent_ema34.iloc[-2] if hasattr(recent_ema34, 'iloc') else recent_ema34[-2]) and \
                 current_price < current_ema34:
                value_zone_analysis['price_behavior'] = 'pullback_bearish'
                value_zone_analysis['analysis'] += ' - Giá vòng xuống qua EMA34'
        
        return value_zone_analysis
    except Exception as e:
        logger.error(f"Lỗi phân tích vùng giá trị MA: {e}")
        return {'zone_type': 'neutral', 'distance_from_ma': 0.0, 'price_behavior': 'normal', 'entry_opportunity': 'none', 'analysis': ''}

def detect_ma_sideways_market(close, ema34, ema89):
    """Phát hiện thị trường đi ngang theo phương pháp trading Việt Nam"""
    try:
        sideways_analysis = {
            'is_sideways': False,
            'sideways_strength': 0.0,
            'recommendation': 'trade',
            'analysis': ''
        }
        
        if len(close) < 20:
            return sideways_analysis
        
        # Lấy 20 giá trị gần nhất
        recent_closes = close.iloc[-20:] if hasattr(close, 'iloc') else close[-20:]
        recent_ema34 = ema34.iloc[-20:] if hasattr(ema34, 'iloc') else ema34[-20:]
        recent_ema89 = ema89.iloc[-20:] if hasattr(ema89, 'iloc') else ema89[-20:]
        
        # Tính độ biến động của EMA34
        ema34_volatility = np.std(recent_ema34) / np.mean(recent_ema34)
        
        # Tính số lần giá cắt qua EMA34
        crossovers = 0
        for i in range(1, len(recent_closes)):
            prev_close = recent_closes.iloc[i-1] if hasattr(recent_closes, 'iloc') else recent_closes[i-1]
            curr_close = recent_closes.iloc[i] if hasattr(recent_closes, 'iloc') else recent_closes[i]
            prev_ema34 = recent_ema34.iloc[i-1] if hasattr(recent_ema34, 'iloc') else recent_ema34[i-1]
            curr_ema34 = recent_ema34.iloc[i] if hasattr(recent_ema34, 'iloc') else recent_ema34[i]
            
            if (prev_close < prev_ema34 and curr_close > curr_ema34) or \
               (prev_close > prev_ema34 and curr_close < curr_ema34):
                crossovers += 1
        
        # Xác định thị trường đi ngang
        if ema34_volatility < 0.02 and crossovers >= 3:  # EMA34 ít biến động và nhiều crossover
            sideways_analysis['is_sideways'] = True
            sideways_analysis['sideways_strength'] = min(crossovers / 10.0, 1.0)
            sideways_analysis['recommendation'] = 'avoid'
            sideways_analysis['analysis'] = f'Thị trường đi ngang - {crossovers} lần cắt EMA34'
        
        return sideways_analysis
    except Exception as e:
        logger.error(f"Lỗi phát hiện thị trường đi ngang: {e}")
        return {'is_sideways': False, 'sideways_strength': 0.0, 'recommendation': 'trade', 'analysis': ''}

def find_support_resistance(highs, lows, current_price):
    """Tìm mức hỗ trợ/kháng cự gần nhất với phân tích nâng cao"""
    try:
        # Chuyển đổi sang pandas Series nếu cần
        if not isinstance(highs, pd.Series):
            highs = pd.Series(highs)
        if not isinstance(lows, pd.Series):
            lows = pd.Series(lows)
        
        # 1. Fibonacci Retracement Levels
        fib_levels = calculate_fibonacci_levels(highs, lows)
        
        # 2. Pivot Points
        pivot_points = calculate_pivot_points(highs, lows, pd.Series([current_price]))
        
        # 3. Dynamic Support/Resistance từ Swing Highs/Lows
        swing_levels = find_swing_levels(highs, lows)
        
        # 4. Volume Weighted Support/Resistance
        volume_levels = find_volume_weighted_levels(highs, lows, pd.Series([current_price]))
        
        # 5. Psychological Levels (round numbers)
        psychological_levels = find_psychological_levels(current_price)
        
        # 6. Historical Support/Resistance từ các đỉnh/đáy quan trọng
        historical_levels = find_historical_levels(highs, lows, current_price)
        
        # Tổng hợp tất cả các mức
        all_support_levels = []
        all_resistance_levels = []
        
        # Thêm Fibonacci levels
        for level_name, price in fib_levels.items():
            if price < current_price:
                all_support_levels.append(('Fibonacci', price, level_name, 0.8))
            else:
                all_resistance_levels.append(('Fibonacci', price, level_name, 0.8))
        
        # Thêm Pivot Points
        for level_name, price in pivot_points.items():
            if price < current_price:
                all_support_levels.append(('Pivot', price, level_name, 0.9))
            else:
                all_resistance_levels.append(('Pivot', price, level_name, 0.9))
        
        # Thêm Swing Levels
        for level in swing_levels:
            if level['type'] == 'support' and level['price'] < current_price:
                all_support_levels.append(('Swing', level['price'], f"Swing Low {level['strength']:.2f}", level['strength']))
            elif level['type'] == 'resistance' and level['price'] > current_price:
                all_resistance_levels.append(('Swing', level['price'], f"Swing High {level['strength']:.2f}", level['strength']))
        
        # Thêm Volume Levels
        for level in volume_levels:
            if level['type'] == 'support' and level['price'] < current_price:
                all_support_levels.append(('Volume', level['price'], f"Volume {level['strength']:.2f}", level['strength']))
            elif level['type'] == 'resistance' and level['price'] > current_price:
                all_resistance_levels.append(('Volume', level['price'], f"Volume {level['strength']:.2f}", level['strength']))
        
        # Thêm Psychological Levels
        for level in psychological_levels:
            if level < current_price:
                all_support_levels.append(('Psychological', level, 'Round Number', 0.7))
            else:
                all_resistance_levels.append(('Psychological', level, 'Round Number', 0.7))
        
        # Thêm Historical Levels
        for level in historical_levels:
            if level['type'] == 'support' and level['price'] < current_price:
                all_support_levels.append(('Historical', level['price'], f"Historical {level['touches']} touches", level['strength']))
            elif level['type'] == 'resistance' and level['price'] > current_price:
                all_resistance_levels.append(('Historical', level['price'], f"Historical {level['touches']} touches", level['strength']))
        
        # Sắp xếp theo khoảng cách và strength
        all_support_levels.sort(key=lambda x: (current_price - x[1], -x[3]))
        all_resistance_levels.sort(key=lambda x: (x[1] - current_price, -x[3]))
        
        # Lấy mức gần nhất và mạnh nhất
        nearest_support = all_support_levels[0] if all_support_levels else (None, min(lows[-20:]), 'Fallback', 0.5)
        nearest_resistance = all_resistance_levels[0] if all_resistance_levels else (None, max(highs[-20:]), 'Fallback', 0.5)
        
        # Tạo kết quả chi tiết
        support_resistance_analysis = {
            'nearest_support': {
                'type': nearest_support[0],
                'timeframe': nearest_support[0],
                'price': nearest_support[1],
                'description': nearest_support[2],
                'strength': nearest_support[3],
                'distance': current_price - nearest_support[1] if nearest_support[1] else 0
            },
            'nearest_resistance': {
                'type': nearest_resistance[0],
                'timeframe': nearest_resistance[0],
                'price': nearest_resistance[1],
                'description': nearest_resistance[2],
                'strength': nearest_resistance[3],
                'distance': nearest_resistance[1] - current_price if nearest_resistance[1] else 0
            },
            'all_support_levels': all_support_levels[:5],  # Top 5
            'all_resistance_levels': all_resistance_levels[:5],  # Top 5
            'fibonacci_levels': fib_levels,
            'pivot_points': pivot_points,
            'swing_levels': swing_levels,
            'volume_levels': volume_levels,
            'psychological_levels': psychological_levels,
            'historical_levels': historical_levels
        }
        
        return nearest_support[1], nearest_resistance[1], support_resistance_analysis
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi tìm support/resistance: {e}")
        # Fallback
        fib_levels = calculate_fibonacci_levels(highs, lows)
        support = min([price for price in fib_levels.values() if price < current_price], default=min(lows[-20:]))
        resistance = max([price for price in fib_levels.values() if price > current_price], default=max(highs[-20:]))
        return support, resistance, None

def calculate_pivot_points(highs, lows, closes):
    """Tính các mức Pivot Points (Classic)"""
    high = highs.iloc[-1] if hasattr(highs, 'iloc') else highs[-1]
    low = lows.iloc[-1] if hasattr(lows, 'iloc') else lows[-1]
    close = closes.iloc[-1] if hasattr(closes, 'iloc') else closes[-1]
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return {'pivot': pivot, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2, 'r3': r3, 's3': s3}

def find_swing_levels(highs, lows, window=20):
    """Tìm các mức Swing High và Swing Low"""
    try:
        swing_levels = []
        
        # Tìm Swing Highs
        for i in range(window, len(highs) - window):
            if all(highs.iloc[i] > highs.iloc[j] for j in range(i-window, i)) and \
               all(highs.iloc[i] > highs.iloc[j] for j in range(i+1, i+window+1)):
                
                # Tính strength dựa trên độ cao và volume
                height = highs.iloc[i] - min(lows.iloc[i-window:i+window])
                strength = min(1.0, height / highs.iloc[i] * 10)  # Normalize strength
                
                swing_levels.append({
                    'type': 'resistance',
                    'price': highs.iloc[i],
                    'position': i,
                    'strength': strength,
                    'height': height
                })
        
        # Tìm Swing Lows
        for i in range(window, len(lows) - window):
            if all(lows.iloc[i] < lows.iloc[j] for j in range(i-window, i)) and \
               all(lows.iloc[i] < lows.iloc[j] for j in range(i+1, i+window+1)):
                
                # Tính strength dựa trên độ sâu và volume
                depth = max(highs.iloc[i-window:i+window]) - lows.iloc[i]
                strength = min(1.0, depth / lows.iloc[i] * 10)  # Normalize strength
                
                swing_levels.append({
                    'type': 'support',
                    'price': lows.iloc[i],
                    'position': i,
                    'strength': strength,
                    'depth': depth
                })
        
        # Sắp xếp theo strength
        swing_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return swing_levels[:10]  # Trả về top 10 swing levels
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi tìm swing levels: {e}")
        return []

def find_volume_weighted_levels(highs, lows, closes, window=50):
    """Tìm các mức support/resistance dựa trên volume"""
    try:
        volume_levels = []
        
        # Tạo price bins
        price_range = max(highs[-window:]) - min(lows[-window:])
        bin_size = price_range / 20  # 20 bins
        
        # Tính volume cho từng bin
        volume_bins = {}
        for i in range(len(highs[-window:])):
            price = (highs.iloc[-window+i] + lows.iloc[-window+i]) / 2
            bin_index = int((price - min(lows[-window:])) / bin_size)
            bin_price = min(lows[-window:]) + bin_index * bin_size
            
            if bin_price not in volume_bins:
                volume_bins[bin_price] = 0
            volume_bins[bin_price] += 1  # Sử dụng count thay vì volume thực
        
        # Tìm các bin có volume cao
        avg_volume = np.mean(list(volume_bins.values()))
        high_volume_bins = [(price, vol) for price, vol in volume_bins.items() if vol > avg_volume * 1.5]
        
        # Chuyển đổi thành support/resistance levels
        for price, volume in high_volume_bins:
            if price < closes.iloc[-1]:  # Support
                volume_levels.append({
                    'type': 'support',
                    'price': price,
                    'strength': min(1.0, volume / avg_volume / 2),
                    'volume_ratio': volume / avg_volume
                })
            else:  # Resistance
                volume_levels.append({
                    'type': 'resistance',
                    'price': price,
                    'strength': min(1.0, volume / avg_volume / 2),
                    'volume_ratio': volume / avg_volume
                })
        
        # Sắp xếp theo strength
        volume_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return volume_levels[:10]  # Trả về top 10 volume levels
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi tìm volume levels: {e}")
        return []

def find_psychological_levels(current_price):
    """Tìm các mức tâm lý (round numbers)"""
    try:
        psychological_levels = []
        
        # Xác định scale dựa trên giá hiện tại
        if current_price >= 1000:  # BTC, ETH
            scale = 1000
            base_levels = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
        elif current_price >= 100:  # Altcoins
            scale = 100
            base_levels = [100, 200, 500, 1000, 2000, 5000]
        elif current_price >= 10:
            scale = 10
            base_levels = [10, 20, 50, 100, 200, 500]
        else:
            scale = 1
            base_levels = [1, 2, 5, 10, 20, 50]
        
        # Tìm các mức gần nhất
        for base in base_levels:
            level = base * scale
            if abs(level - current_price) / current_price < 0.5:  # Trong vòng 50%
                psychological_levels.append(level)
        
        # Thêm các mức 0.5, 0.25, 0.75
        for multiplier in [0.25, 0.5, 0.75]:
            for base in base_levels:
                level = base * scale * multiplier
                if abs(level - current_price) / current_price < 0.5:
                    psychological_levels.append(level)
        
        return sorted(list(set(psychological_levels)))  # Loại bỏ duplicates
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi tìm psychological levels: {e}")
        return []

def find_historical_levels(highs, lows, current_price, lookback=100):
    """Tìm các mức support/resistance lịch sử quan trọng"""
    try:
        historical_levels = []
        
        # Tìm các đỉnh và đáy quan trọng
        peaks = []
        troughs = []
        
        for i in range(1, len(highs[-lookback:]) - 1):
            if highs.iloc[-lookback+i] > highs.iloc[-lookback+i-1] and \
               highs.iloc[-lookback+i] > highs.iloc[-lookback+i+1]:
                peaks.append(highs.iloc[-lookback+i])
            
            if lows.iloc[-lookback+i] < lows.iloc[-lookback+i-1] and \
               lows.iloc[-lookback+i] < lows.iloc[-lookback+i+1]:
                troughs.append(lows.iloc[-lookback+i])
        
        # Nhóm các mức gần nhau (cluster analysis)
        def cluster_levels(levels, tolerance=0.02):
            if not levels:
                return []
            
            clusters = []
            sorted_levels = sorted(levels)
            
            current_cluster = [sorted_levels[0]]
            for level in sorted_levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                    current_cluster.append(level)
                else:
                    # Tính trung bình của cluster
                    avg_level = np.mean(current_cluster)
                    clusters.append({
                        'price': avg_level,
                        'touches': len(current_cluster),
                        'strength': min(1.0, len(current_cluster) / 5)  # Normalize strength
                    })
                    current_cluster = [level]
            
            # Xử lý cluster cuối cùng
            if current_cluster:
                avg_level = np.mean(current_cluster)
                clusters.append({
                    'price': avg_level,
                    'touches': len(current_cluster),
                    'strength': min(1.0, len(current_cluster) / 5)
                })
            
            return clusters
        
        # Tạo clusters cho peaks và troughs
        resistance_clusters = cluster_levels(peaks)
        support_clusters = cluster_levels(troughs)
        
        # Thêm vào historical levels
        for cluster in resistance_clusters:
            if cluster['price'] > current_price:
                historical_levels.append({
                    'type': 'resistance',
                    'price': cluster['price'],
                    'touches': cluster['touches'],
                    'strength': cluster['strength']
                })
        
        for cluster in support_clusters:
            if cluster['price'] < current_price:
                historical_levels.append({
                    'type': 'support',
                    'price': cluster['price'],
                    'touches': cluster['touches'],
                    'strength': cluster['strength']
                })
        
        # Sắp xếp theo strength và số lần chạm
        historical_levels.sort(key=lambda x: (x['strength'], x['touches']), reverse=True)
        
        return historical_levels[:15]  # Trả về top 15 historical levels
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi tìm historical levels: {e}")
        return []

def analyze_support_resistance_strength(support_resistance_analysis, current_price):
    """Phân tích độ mạnh của các mức support/resistance"""
    try:
        if not support_resistance_analysis:
            return None
        
        analysis = {
            'support_analysis': {},
            'resistance_analysis': {},
            'breakout_potential': {},
            'consolidation_zones': [],
            'recommendations': []
        }
        
        # Phân tích Support
        support = support_resistance_analysis['nearest_support']
        if support and support['price']:
            support_distance = support['distance']
            support_strength = support['strength']
            
            if support_distance < current_price * 0.01:  # < 1%
                analysis['support_analysis']['status'] = 'Very Close'
                analysis['support_analysis']['risk'] = 'High - Price near support'
                analysis['support_analysis']['action'] = 'Watch for bounce or breakdown'
            elif support_distance < current_price * 0.05:  # < 5%
                analysis['support_analysis']['status'] = 'Close'
                analysis['support_analysis']['risk'] = 'Medium - Price approaching support'
                analysis['support_analysis']['action'] = 'Prepare for potential bounce'
            else:
                analysis['support_analysis']['status'] = 'Safe Distance'
                analysis['support_analysis']['risk'] = 'Low - Price far from support'
                analysis['support_analysis']['action'] = 'Support not immediate concern'
            
            analysis['support_analysis']['price'] = support['price']
            analysis['support_analysis']['strength'] = support_strength
            analysis['support_analysis']['type'] = support['type']
            analysis['support_analysis']['description'] = support['description']
        
        # Phân tích Resistance
        resistance = support_resistance_analysis['nearest_resistance']
        if resistance and resistance['price']:
            resistance_distance = resistance['distance']
            resistance_strength = resistance['strength']
            
            if resistance_distance < current_price * 0.01:  # < 1%
                analysis['resistance_analysis']['status'] = 'Very Close'
                analysis['resistance_analysis']['risk'] = 'High - Price near resistance'
                analysis['resistance_analysis']['action'] = 'Watch for breakout or rejection'
            elif resistance_distance < current_price * 0.05:  # < 5%
                analysis['resistance_analysis']['status'] = 'Close'
                analysis['resistance_analysis']['risk'] = 'Medium - Price approaching resistance'
                analysis['resistance_analysis']['action'] = 'Prepare for potential breakout'
            else:
                analysis['resistance_analysis']['status'] = 'Safe Distance'
                analysis['resistance_analysis']['risk'] = 'Low - Price far from resistance'
                analysis['resistance_analysis']['action'] = 'Resistance not immediate concern'
            
            analysis['resistance_analysis']['price'] = resistance['price']
            analysis['resistance_analysis']['strength'] = resistance_strength
            analysis['resistance_analysis']['type'] = resistance['type']
            analysis['resistance_analysis']['description'] = resistance['description']
        
        # Phân tích tiềm năng breakout
        if support and resistance:
            range_size = resistance['price'] - support['price']
            current_position = (current_price - support['price']) / range_size
            
            if current_position < 0.2:  # Gần support
                analysis['breakout_potential']['direction'] = 'Downside'
                analysis['breakout_potential']['probability'] = 'Medium-High'
                analysis['breakout_potential']['target'] = support['price'] * 0.95
                analysis['breakout_potential']['support_price'] = support['price']
                analysis['breakout_potential']['resistance_price'] = resistance['price']
                analysis['recommendations'].append('Watch for support breakdown - prepare for short')
            elif current_position > 0.8:  # Gần resistance
                analysis['breakout_potential']['direction'] = 'Upside'
                analysis['breakout_potential']['probability'] = 'Medium-High'
                analysis['breakout_potential']['target'] = resistance['price'] * 1.05
                analysis['breakout_potential']['support_price'] = support['price']
                analysis['breakout_potential']['resistance_price'] = resistance['price']
                analysis['recommendations'].append('Watch for resistance breakout - prepare for long')
            else:  # Ở giữa range
                analysis['breakout_potential']['direction'] = 'Sideways'
                analysis['breakout_potential']['probability'] = 'Low'
                analysis['breakout_potential']['target'] = 'Range bound trading'
                analysis['breakout_potential']['support_price'] = support['price']
                analysis['breakout_potential']['resistance_price'] = resistance['price']
                analysis['recommendations'].append('Range bound market - trade between support/resistance')
        
        # Tìm các vùng consolidation
        if support and resistance:
            consolidation_range = {
                'support': support['price'],
                'resistance': resistance['price'],
                'range_size': resistance['price'] - support['price'],
                'range_percentage': (resistance['price'] - support['price']) / current_price * 100
            }
            analysis['consolidation_zones'].append(consolidation_range)
        
        # Thêm recommendations dựa trên strength
        if support and support['strength'] > 0.8:
            analysis['recommendations'].append(f'Strong support at ${support["price"]:.4f} - High probability bounce')
        elif support and support['strength'] < 0.4:
            analysis['recommendations'].append(f'Weak support at ${support["price"]:.4f} - Low probability bounce')
        
        if resistance and resistance['strength'] > 0.8:
            analysis['recommendations'].append(f'Strong resistance at ${resistance["price"]:.4f} - High probability rejection')
        elif resistance and resistance['strength'] < 0.4:
            analysis['recommendations'].append(f'Weak resistance at ${resistance["price"]:.4f} - Low probability rejection')
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi phân tích support/resistance strength: {e}")
        return None

def calculate_volume_profile(highs, lows, volumes, bins=10):
    """Tính Volume Profile đơn giản (phân bố khối lượng theo mức giá)"""
    # Chuyển đổi sang list nếu là pandas Series
    highs_list = highs.tolist() if hasattr(highs, 'tolist') else highs
    lows_list = lows.tolist() if hasattr(lows, 'tolist') else lows
    volumes_list = volumes.tolist() if hasattr(volumes, 'tolist') else volumes
    
    price_range = max(highs_list[-50:]) - min(lows_list[-50:])
    bin_size = price_range / bins
    volume_bins = [0] * bins
    for i in range(len(highs_list[-50:])):
        price = (highs_list[-50:][i] + lows_list[-50:][i]) / 2
        bin_index = min(int((price - min(lows_list[-50:])) / bin_size), bins - 1)
        volume_bins[bin_index] += volumes_list[-50:][i]
    max_volume_price = min(lows_list[-50:]) + volume_bins.index(max(volume_bins)) * bin_size
    return max_volume_price

def calculate_vwap(highs, lows, closes, volumes):
    """Tính VWAP (Volume Weighted Average Price)"""
    # Chuyển đổi sang numpy array nếu cần
    highs_array = highs.values if hasattr(highs, 'values') else np.array(highs)
    lows_array = lows.values if hasattr(lows, 'values') else np.array(lows)
    closes_array = closes.values if hasattr(closes, 'values') else np.array(closes)
    volumes_array = volumes.values if hasattr(volumes, 'values') else np.array(volumes)
    
    typical_prices = (highs_array[-20:] + lows_array[-20:] + closes_array[-20:]) / 3
    vwap = np.sum(typical_prices * volumes_array[-20:]) / np.sum(volumes_array[-20:])
    return vwap

def detect_price_patterns(highs, lows, closes):
    """Phát hiện các mô hình giá"""
    pattern = 'None'
    
    # Chuyển đổi sang list nếu là pandas Series
    highs_list = highs.tolist() if hasattr(highs, 'tolist') else highs
    lows_list = lows.tolist() if hasattr(lows, 'tolist') else lows
    closes_list = closes.tolist() if hasattr(closes, 'tolist') else closes
    
    # Đỉnh đầu vai (Head and Shoulders)
    if len(highs_list) >= 7:
        left_shoulder = highs_list[-5] > highs_list[-6] and highs_list[-5] > highs_list[-4]
        head = highs_list[-3] > highs_list[-5] and highs_list[-3] > highs_list[-1]
        right_shoulder = highs_list[-1] > highs_list[-2] and highs_list[-1] < highs_list[-3]
        if left_shoulder and head and right_shoulder:
            pattern = 'Head and Shoulders'
    
    # Đỉnh đôi (Double Top)
    elif len(highs_list) >= 5:
        if abs(highs_list[-3] - highs_list[-1]) / highs_list[-3] < 0.01 and highs_list[-3] > highs_list[-2] and highs_list[-1] > highs_list[-2]:
            pattern = 'Double Top'
    
    # Cờ (Flag)
    elif len(highs_list) >= 10:
        uptrend = all(closes_list[i] > closes_list[i-1] for i in range(-10, -5))
        consolidation = max(highs_list[-5:]) - min(lows_list[-5:]) < 0.02 * closes_list[-1]
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
    
    # Chuyển đổi sang list nếu là pandas Series
    opens_list = opens.tolist() if hasattr(opens, 'tolist') else opens
    highs_list = highs.tolist() if hasattr(highs, 'tolist') else highs
    lows_list = lows.tolist() if hasattr(lows, 'tolist') else lows
    closes_list = closes.tolist() if hasattr(closes, 'tolist') else closes
    
    # Nến đơn
    body_size = abs(opens_list[-1] - closes_list[-1])
    candle_range = highs_list[-1] - lows_list[-1]
    if candle_range > 0:
        # Doji
        if body_size / candle_range < 0.1:
            patterns.append('Doji')
        # Hammer
        if (closes_list[-1] > opens_list[-1] and 
            (highs_list[-1] - closes_list[-1]) / candle_range < 0.2 and 
            (opens_list[-1] - lows_list[-1]) / candle_range > 0.6):
            patterns.append('Hammer')
        # Shooting Star
        if (closes_list[-1] < opens_list[-1] and 
            (highs_list[-1] - opens_list[-1]) / candle_range > 0.6 and 
            (closes_list[-1] - lows_list[-1]) / candle_range < 0.2):
            patterns.append('Shooting Star')
        # Spinning Top
        if body_size / candle_range < 0.3 and (highs_list[-1] - closes_list[-1]) / candle_range > 0.3 and (opens_list[-1] - lows_list[-1]) / candle_range > 0.3:
            patterns.append('Spinning Top')

    # Nến đôi
    if len(opens_list) >= 2:
        # Bullish Engulfing
        if (closes_list[-2] < opens_list[-2] and closes_list[-1] > opens_list[-1] and 
            closes_list[-1] > opens_list[-2] and opens_list[-1] < closes_list[-2]):
            patterns.append('Bullish Engulfing')
        # Bearish Engulfing
        if (closes_list[-2] > opens_list[-2] and closes_list[-1] < opens_list[-1] and 
            closes_list[-1] < opens_list[-2] and opens_list[-1] > closes_list[-2]):
            patterns.append('Bearish Engulfing')

    # Nến ba
    if len(opens_list) >= 3:
        # Morning Star
        if (closes_list[-3] < opens_list[-3] and 
            abs(closes_list[-2] - opens_list[-2]) / (highs_list[-2] - lows_list[-2]) < 0.3 and 
            closes_list[-1] > opens_list[-1] and closes_list[-1] > (highs_list[-3] + lows_list[-3]) / 2):
            patterns.append('Morning Star')
        # Evening Star
        if (closes_list[-3] > opens_list[-3] and 
            abs(closes_list[-2] - opens_list[-2]) / (highs_list[-2] - lows_list[-2]) < 0.3 and 
            closes_list[-1] < opens_list[-1] and closes_list[-1] < (highs_list[-3] + lows_list[-3]) / 2):
            patterns.append('Evening Star')
        # Three White Soldiers
        if (all(closes_list[i] > opens_list[i] and closes_list[i] > closes_list[i-1] for i in [-3, -2, -1]) and
            all((highs_list[i] - closes_list[i]) / (highs_list[i] - lows_list[i]) < 0.2 for i in [-3, -2, -1])):
            patterns.append('Three White Soldiers')
        # Three Black Crows
        if (all(closes_list[i] < opens_list[i] and closes_list[i] < closes_list[i-1] for i in [-3, -2, -1]) and
            all((closes_list[i] - lows_list[i]) / (highs_list[i] - lows_list[i]) < 0.2 for i in [-3, -2, -1])):
            patterns.append('Three Black Crows')

    return patterns

def detect_elliott_wave(highs, lows, closes):
    """Phát hiện mô hình Elliott Wave đơn giản"""
    wave_pattern = 'None'
    
    # Chuyển đổi sang list nếu là pandas Series
    closes_list = closes.tolist() if hasattr(closes, 'tolist') else closes
    
    if len(closes_list) >= 10:
        # Tìm 5 sóng tăng (Wave 1-5)
        waves = []
        current_wave = 0
        wave_start = 0
        
        for i in range(1, len(closes_list)):
            if closes_list[i] > closes_list[i-1]:  # Sóng tăng
                if current_wave == 0 or current_wave % 2 == 0:  # Bắt đầu sóng mới
                    current_wave += 1
                    wave_start = i-1
            elif closes_list[i] < closes_list[i-1]:  # Sóng giảm
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
    """Phân tích kỹ thuật tối ưu với 12 chỉ số cốt lõi, ML và phân tích hội tụ"""
    # Khởi tạo commodity_signals để tránh lỗi NameError
    commodity_signals = {}
    
    # Chuyển đổi numpy array sang pandas Series để tương thích với thư viện ta
    close = pd.Series(data['close'])
    high = pd.Series(data['high'])
    low = pd.Series(data['low'])
    volume = pd.Series(data['volume'])
    open = pd.Series(data['open'])
    
    # Helper function để lấy giá trị cuối cùng
    def get_last(series):
        return series.iloc[-1] if hasattr(series, 'iloc') else series[-1]
    
    def get_last_n(series, n):
        if hasattr(series, 'iloc'):
            return series.iloc[-n:].tolist()
        else:
            return series[-n:]

    # === 1. TREND INDICATORS (5 chỉ số cốt lõi) ===
    ema34 = ta.trend.ema_indicator(close, window=34)              # Trend ngắn hạn
    ema34 = ta.trend.ema_indicator(close, window=34)              # Elliott Wave main waves
    ema50 = ta.trend.ema_indicator(close, window=50)              # Trend trung hạn  
    ema89 = ta.trend.ema_indicator(close, window=89)              # Elliott Wave main waves
    adx = ta.trend.adx(high, low, close, window=7)      # Trend strength

    # === 2. MOMENTUM INDICATORS (3 chỉ số cốt lõi) ===
    rsi = ta.momentum.rsi(close, window=7)                 # Momentum chuẩn
    stoch_k = ta.momentum.stoch(high, low, close, window=7)  # Stochastic %K
    stoch_d = ta.momentum.stoch_signal(high, low, close, window=7)  # Stochastic %D
    macd_line = ta.trend.macd(close, window_fast=6, window_slow=13)  # MACD line
    macd_signal = ta.trend.macd_signal(close, window_fast=6, window_slow=13, window_sign=4)  # MACD signal
    macd_hist = ta.trend.macd_diff(close, window_fast=6, window_slow=13, window_sign=4)  # MACD histogram

    # === 3. VOLATILITY INDICATORS (2 chỉ số cốt lõi) ===
    bb_upper = ta.volatility.bollinger_hband(close, window=10, window_dev=2)       # BB Upper
    bb_middle = ta.volatility.bollinger_mavg(close, window=10)       # BB Middle
    bb_lower = ta.volatility.bollinger_lband(close, window=10, window_dev=2)       # BB Lower
    atr = ta.volatility.average_true_range(high, low, close, window=7)      # Volatility thuần

    # === 4. VOLUME INDICATORS (2 chỉ số cốt lõi) ===
    obv = ta.volume.on_balance_volume(close, volume)                   # Volume flow
    vwap = calculate_vwap(high, low, close, volume) # Volume price level

    # === 5. SUPPORT/RESISTANCE (2 chỉ số cốt lõi) ===
    pivot_points = calculate_pivot_points(high, low, close)
    support, resistance, support_resistance_analysis = find_support_resistance(high, low, current_price)
    
    # Phân tích độ mạnh của support/resistance
    sr_strength_analysis = analyze_support_resistance_strength(support_resistance_analysis, current_price)

    # === 6. MÔ HÌNH GIÁ VÀ NẾN ===
    price_pattern = detect_price_patterns(high, low, close)
    candlestick_patterns = detect_candlestick_patterns(open, high, low, close)
    
    # === 7. SMART MONEY CONCEPTS ===
    order_blocks = detect_order_blocks(high, low, close, volume)
    fvgs = detect_fair_value_gaps(high, low, close)
    liquidity_zones = detect_liquidity_zones(high, low, close, volume)
    mitigation_zones = detect_mitigation_zones(high, low, close)
    
    # Phân tích SMC và Price Action
    smc_signals = analyze_smc_signals(current_price, order_blocks, fvgs, liquidity_zones, mitigation_zones)
    
    # === 8. PHÂN TÍCH EMA 34/89 THEO ELLIOTT WAVE ===
    ema_trend_analysis = analyze_ema_34_89_trend(close, ema34, ema89, current_price)
    ema_breakout_pattern = detect_ema_breakout_pattern(close, ema34, ema89)
    ma_value_zones = analyze_ma_value_zones(close, ema34, ema89, current_price)
    sideways_market = detect_ma_sideways_market(close, ema34, ema89)
    
    # === 9. PHÂN TÍCH DIVERGENCE/CONVERGENCE - TRỌNG SỐ CAO ===
    divergences = analyze_all_divergences(close, rsi, macd_line, volume)
    divergence_consensus = calculate_divergence_consensus(divergences)
    
    # === 10. PHÂN TÍCH ĐA KHUNG THỜI GIAN EMA ===
    multi_tf_analysis = analyze_multi_timeframe_ema_system(symbol, current_price) if symbol else {'entry_signal': 'hold', 'trend_alignment': False, 'analysis': ''}
    
    # === 11. PHÂN TÍCH SMART MONEY ===
    smart_money_analysis = detect_smart_money_accumulation_distribution(close, volume, ema34, ema89)
    whale_analysis = detect_whale_activity(close, volume, ema34, ema89)
    
    # === 12. PHÂN TÍCH TÂM LÝ THỊ TRƯỜNG ===
    fib_levels = calculate_fibonacci_levels(high, low)
    fib_psychology = analyze_fibonacci_psychology(current_price, fib_levels, max(high[-50:]) - min(low[-50:]))
    
    # === 13. TÍNH TOÁN TÍN HIỆU CƠ BẢN ===
    
    # RSI Signal
    rsi_signal = 'Hold'
    if get_last(rsi) < 25:
        rsi_signal = 'Long'
    elif get_last(rsi) > 75:
        rsi_signal = 'Short'
    
    # Stochastic Signal
    stoch_signal = 'Hold'
    if get_last(stoch_k) < 20 and get_last(stoch_k) > get_last(stoch_d):
        stoch_signal = 'Long'
    elif get_last(stoch_k) > 80 and get_last(stoch_k) < get_last(stoch_d):
        stoch_signal = 'Short'

    # MACD Signal
    macd_signal = 'Hold'
    try:
        macd_line_last_2 = get_last_n(macd_line, 2)
        macd_signal_last_2 = get_last_n(macd_signal, 2)
        if (not np.isnan(macd_line_last_2[0]) and not np.isnan(macd_signal_last_2[0]) and
            macd_line_last_2[0] < macd_signal_last_2[0] and get_last(macd_line) > get_last(macd_signal)):
            macd_signal = 'Long'
        elif (not np.isnan(macd_line_last_2[0]) and not np.isnan(macd_signal_last_2[0]) and
              macd_line_last_2[0] > macd_signal_last_2[0] and get_last(macd_line) < get_last(macd_signal)):
            macd_signal = 'Short'
    except:
        pass

    # MA Signal
    ma_signal = 'Hold'
    ma_distance = abs(get_last(ema34) - get_last(ema89)) / get_last(ema89)
    if get_last(ema34) > get_last(ema89) and ma_distance > 0.01:
        ma_signal = 'Long'
    elif get_last(ema34) < get_last(ema89) and ma_distance > 0.01:
        ma_signal = 'Short'

    # EMA 34/89 Signal (Elliott Wave based)
    ema_34_89_signal = 'Hold'
    if ema_trend_analysis['entry_signal'] == 'buy':
        ema_34_89_signal = 'Long'
    elif ema_trend_analysis['entry_signal'] == 'sell':
        ema_34_89_signal = 'Short'
    
    # EMA Breakout Signal
    ema_breakout_signal = 'Hold'
    if ema_breakout_pattern['signal'] == 'buy':
        ema_breakout_signal = 'Long'
    elif ema_breakout_pattern['signal'] == 'sell':
        ema_breakout_signal = 'Short'
    
    # Value Zone Signal
    value_zone_signal = 'Hold'
    if ma_value_zones['entry_opportunity'] == 'high' and ma_value_zones['price_behavior'] in ['pullback_bullish', 'pullback_bearish']:
        if ma_value_zones['price_behavior'] == 'pullback_bullish':
            value_zone_signal = 'Long'
        else:
            value_zone_signal = 'Short'
    
    # Sideways Market Signal
    sideways_signal = 'Hold'
    if sideways_market['recommendation'] == 'avoid':
        sideways_signal = 'Avoid'  # Special signal to avoid trading
    
    # Multi-timeframe EMA Signal
    multi_tf_signal = 'Hold'
    if multi_tf_analysis['entry_signal'] == 'buy':
        multi_tf_signal = 'Long'
    elif multi_tf_analysis['entry_signal'] == 'sell':
        multi_tf_signal = 'Short'
    
    # Smart Money Signal
    smart_money_signal = 'Hold'
    if smart_money_analysis['smart_money_signal'] == 'buy':
        smart_money_signal = 'Long'
    elif smart_money_analysis['smart_money_signal'] == 'sell':
        smart_money_signal = 'Short'
    
    # Whale Activity Signal
    whale_signal = 'Hold'
    if whale_analysis['whale_signal'] == 'watch':
        whale_signal = 'Watch'  # Special signal to watch for whale activity
    
    # Market Psychology Signal (Fibonacci-based)
    psychology_signal = 'Hold'
    if fib_psychology['market_sentiment'] in ['strong_bullish', 'bullish']:
        psychology_signal = 'Long'
    elif fib_psychology['market_sentiment'] in ['strong_bearish', 'bearish']:
        psychology_signal = 'Short'

    # ADX Signal
    adx_signal = 'Hold'
    if get_last(adx) > 25:
        if get_last(close) > get_last(ema34):
            adx_signal = 'Long'
        elif get_last(close) < get_last(ema34):
            adx_signal = 'Short'

    # Bollinger Bands Signal
    bb_signal = 'Hold'
    if current_price <= get_last(bb_lower) * 0.995:
        bb_signal = 'Long'
    elif current_price >= get_last(bb_upper) * 1.005:
        bb_signal = 'Short'

    # OBV Signal
    obv_signal = 'Hold'
    obv_slope = get_last(obv) - get_last_n(obv, 10)[0]
    obv_change = obv_slope / get_last_n(obv, 10)[0] if get_last_n(obv, 10)[0] != 0 else 0
    if obv_change > 0.05 and get_last(close) > get_last(ema50):
        obv_signal = 'Long'
    elif obv_change < -0.05 and get_last(close) < get_last(ema50):
        obv_signal = 'Short'

    # VWAP Signal
    vwap_signal = 'Hold'
    vwap_distance = abs(current_price - vwap) / vwap
    if current_price > vwap and vwap_distance > 0.02:
        vwap_signal = 'Long'
    elif current_price < vwap and vwap_distance > 0.02:
        vwap_signal = 'Short'

    # ATR Signal
    atr_signal = 'Hold'
    atr_avg = np.mean(get_last_n(atr, 10))
    if get_last(atr) > atr_avg * 1.5:
        if get_last(close) > get_last(ema50):
            atr_signal = 'Long'
        elif get_last(close) < get_last(ema50):
            atr_signal = 'Short'

    # Pivot Points Signal
    pivot_signal = 'Hold'
    pivot_distance = min(abs(current_price - pivot_points['s1']), abs(current_price - pivot_points['r1'])) / current_price
    if current_price < pivot_points['s1'] and pivot_distance > 0.01:
        pivot_signal = 'Long'
    elif current_price > pivot_points['r1'] and pivot_distance > 0.01:
        pivot_signal = 'Short'
    
    # Support/Resistance Signal
    sr_signal = 'Hold'
    support_distance = (current_price - support) / current_price
    resistance_distance = (resistance - current_price) / current_price
    
    if support_distance < 0.01 and current_price > support:  # Gần support
        sr_signal = 'Long'
    elif resistance_distance < 0.01 and current_price < resistance:  # Gần resistance
        sr_signal = 'Short'

    # Candlestick Signal
    candlestick_signal = 'Hold'
    if any(p in ['Hammer', 'Bullish Engulfing', 'Morning Star', 'Three White Soldiers'] for p in candlestick_patterns):
        candlestick_signal = 'Long'
    elif any(p in ['Shooting Star', 'Bearish Engulfing', 'Evening Star', 'Three Black Crows'] for p in candlestick_patterns):
        candlestick_signal = 'Short'

    # Price Pattern Signal
    price_pattern_signal = 'Hold'
    if price_pattern in ['Head and Shoulders', 'Double Top']:
        price_pattern_signal = 'Short'
    elif price_pattern == 'Flag' and get_last(close) > get_last(ema50):
        price_pattern_signal = 'Long'

    # === 10. TÍN HIỆU ĐẶC BIỆT CHO HÀNG HÓA ===
    # Đã loại bỏ code liên quan đến vàng và dầu

    # === 11. TẠO DANH SÁCH TÍN HIỆU CƠ BẢN ===
    basic_signals = [
        rsi_signal, stoch_signal, macd_signal, ma_signal, adx_signal,
        bb_signal, obv_signal, vwap_signal, atr_signal, pivot_signal,
        candlestick_signal, price_pattern_signal, sr_signal,
        ema_34_89_signal, ema_breakout_signal, value_zone_signal, multi_tf_signal,  # EMA 34/89 signals
        smart_money_signal, whale_signal, psychology_signal,  # Smart money & psychology signals
        smc_signals['order_block_signal'], smc_signals['fvg_signal'], 
        smc_signals['liquidity_signal'], smc_signals['mitigation_signal']
    ]
    
    # Thêm một số tín hiệu cơ bản dựa trên xu hướng giá
    price_trend_signal = 'Hold'
    if len(close) >= 5:
        recent_trend = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
        if recent_trend > 0.02:  # Tăng > 2%
            price_trend_signal = 'Long'
        elif recent_trend < -0.02:  # Giảm > 2%
            price_trend_signal = 'Short'
    
    # Thêm tín hiệu dựa trên volume
    volume_signal = 'Hold'
    if len(volume) >= 10:
        current_volume = volume.iloc[-1]
        avg_volume = volume.iloc[-10:].mean()
        if current_volume > avg_volume * 1.5:  # Volume cao
            if get_last(close) > get_last(ema34):
                volume_signal = 'Long'
            else:
                volume_signal = 'Short'
    
    # Thêm các tín hiệu mới vào danh sách
    basic_signals.extend([price_trend_signal, volume_signal])
    
    # Debug logging cho basic signals
    

    # === 12. XỬ LÝ DIVERGENCE VỚI TRỌNG SỐ CAO ===
    divergence_signal = divergence_consensus['signal']
    divergence_strength = divergence_consensus['strength']
    divergence_count = divergence_consensus['count']
    
    # Debug logging cho divergence
    
    
    # Tạo danh sách tín hiệu cuối cùng với trọng số divergence
    final_signals = basic_signals.copy()
    
    # NẠNG CAO TRỌNG SỐ CHO DIVERGENCE
    if divergence_signal != 'Hold' and divergence_strength > 0.2:
        # Thêm divergence signal nhiều lần dựa trên strength
        divergence_weight = int(divergence_strength * 10)  # Tăng từ 5 lên 10
        for _ in range(divergence_weight):
            final_signals.append(divergence_signal)
        
        # Thêm cảnh báo đặc biệt cho divergence mạnh
        if divergence_strength > 0.5:
            # Thêm thêm 5 lần nữa cho divergence rất mạnh
            for _ in range(5):
                final_signals.append(divergence_signal)
        


    # === 13. TÍN HIỆU CỰC MẠNH (EXTRA WEIGHT) ===
    extra_signals = []
    
    # RSI cực mạnh
    if get_last(rsi) < 20:
        extra_signals.extend(['Long', 'Long', 'Long'])
    elif get_last(rsi) > 80:
        extra_signals.extend(['Short', 'Short', 'Short'])

    
    # Stochastic cực mạnh
    if get_last(stoch_k) < 10:
        extra_signals.extend(['Long', 'Long', 'Long'])
    elif get_last(stoch_k) > 90:
        extra_signals.extend(['Short', 'Short', 'Short'])

    
    # Bollinger Bands breakout mạnh
    if current_price < get_last(bb_lower) * 0.985:
        extra_signals.extend(['Long', 'Long', 'Long'])

    elif current_price > get_last(bb_upper) * 1.015:
        extra_signals.extend(['Short', 'Short', 'Short'])

    
    # MACD crossover mạnh
    try:
        if (get_last(macd_line) > get_last(macd_signal) * 1.2 and 
            get_last_n(macd_line, 2)[0] <= get_last_n(macd_signal, 2)[0]):
            extra_signals.extend(['Long', 'Long', 'Long'])

        elif (get_last(macd_line) < get_last(macd_signal) * 0.8 and 
              get_last_n(macd_line, 2)[0] >= get_last_n(macd_signal, 2)[0]):
            extra_signals.extend(['Short', 'Short', 'Short'])

    except:
        pass
    

    
    # Thêm một số tín hiệu extra dựa trên điều kiện thị trường
    market_condition_signals = []
    
    # Tín hiệu dựa trên volatility
    if get_last(atr) > np.mean([atr.iloc[i] for i in range(-10, 0)]) * 1.2:
        if get_last(close) > get_last(ema34):
            market_condition_signals.extend(['Long', 'Long'])
        else:
            market_condition_signals.extend(['Short', 'Short'])

    
    # Tín hiệu dựa trên momentum
    if len(close) >= 3:
        momentum = (close.iloc[-1] - close.iloc[-3]) / close.iloc[-3]
        if abs(momentum) > 0.03:  # Momentum > 3%
            if momentum > 0:
                market_condition_signals.extend(['Long', 'Long'])
            else:
                market_condition_signals.extend(['Short', 'Short'])

    
    extra_signals.extend(market_condition_signals)


    # === 16. MACHINE LEARNING PREDICTION ===
    ml_prediction = None
    try:
        # Chỉ sử dụng ML cho các timeframe đã được train
        if timeframe in ML_TIMEFRAMES:
            ml_prediction = predict_with_ml(symbol, timeframe, data)
            if ml_prediction and ml_prediction['confidence'] > ML_CONFIDENCE_THRESHOLD:
                # Thêm tín hiệu ML vào danh sách với trọng số cao
                ml_weight = int(ml_prediction['confidence'] * 5)  # Trọng số dựa trên confidence
                for _ in range(ml_weight):
                    final_signals.append(ml_prediction['signal'])
                logger.info(f"🤖 ML {ml_prediction['model_name']}: {ml_prediction['signal']} (Confidence: {ml_prediction['confidence']:.3f})")

        else:
            logger.debug(f"⏭️ Bỏ qua ML prediction cho {symbol} ({timeframe}) - không có model được train")
    except Exception as e:
        logger.warning(f"⚠️ Lỗi ML prediction cho {symbol}: {e}")

    # === 17. PHÂN TÍCH HỘI TỤ (CONVERGENCE ANALYSIS) ===
    convergence_analysis = None
    try:
        convergence_analysis = analyze_convergence(data)
        if convergence_analysis and convergence_analysis['overall_convergence'] > CONVERGENCE_THRESHOLD:
            # Thêm tín hiệu convergence vào danh sách
            convergence_weight = int(convergence_analysis['strength'] * 3)
            for signal_info in convergence_analysis['signals']:
                for _ in range(convergence_weight):
                    final_signals.append(signal_info['signal'])
            logger.info(f"🎯 Convergence Analysis: {convergence_analysis['overall_convergence']:.3f} - {len(convergence_analysis['signals'])} signals")

    except Exception as e:
        logger.warning(f"⚠️ Lỗi convergence analysis cho {symbol}: {e}")

    # === 18. TÍNH TOÁN CONSENSUS CUỐI CÙNG ===
    all_signals = final_signals + extra_signals
    
    long_count = all_signals.count('Long')
    short_count = all_signals.count('Short')
    hold_count = all_signals.count('Hold')
    
    total_signals = len(all_signals)
    

    
    # Debug: Hiển thị chi tiết các tín hiệu
    if len(all_signals) > 0:
        signal_counts = {}
        for signal in all_signals:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1

    
    if total_signals == 0:
        consensus = 'Hold'
        confidence = 0.0
    else:
        if long_count > short_count:
            consensus = 'Long'
            confidence = long_count / total_signals
        elif short_count > long_count:
            consensus = 'Short'
            confidence = short_count / total_signals
        else:
            consensus = 'Hold'
            confidence = 0.5
    
    # === 15. TÍNH TOÁN ĐIỂM ENTRY ===
    entry_points = calculate_entry_points(current_price, high, low, close, rsi, bb_upper, bb_lower, ema50, pivot_points, support, resistance, support_resistance_analysis, consensus)

    # === 19. TRẢ VỀ KẾT QUẢ TỐI ƯU ===
    return {
        'trend': 'bullish' if consensus == 'Long' else 'bearish' if consensus == 'Short' else 'neutral',
        'signal': consensus,
        'confidence': confidence,
        'consensus_ratio': confidence,  # Thêm consensus_ratio để tương thích
        'strength': divergence_strength,
        'timeframe': timeframe,  # Thêm timeframe vào kết quả
        'indicators': {
            'rsi': get_last(rsi),
            'stoch_k': get_last(stoch_k),
            'stoch_d': get_last(stoch_d),
            'macd_line': get_last(macd_line),
            'macd_signal': get_last(macd_signal),
            'ema34': get_last(ema34),
            'ema50': get_last(ema50),
            'adx': get_last(adx),
            'bb_upper': get_last(bb_upper),
            'bb_middle': get_last(bb_middle),
            'bb_lower': get_last(bb_lower),
            'atr': get_last(atr),
            'obv': get_last(obv),
            'vwap': vwap,
            'support': support,
            'resistance': resistance
        },
        'support_resistance_analysis': support_resistance_analysis,
        'sr_strength_analysis': sr_strength_analysis,
        'signals': {
            'rsi': rsi_signal,
            'stoch': stoch_signal,
            'macd': macd_signal,
            'ma': ma_signal,
            'adx': adx_signal,
            'bb': bb_signal,
            'obv': obv_signal,
            'vwap': vwap_signal,
            'atr': atr_signal,
            'pivot': pivot_signal,
            'candlestick': candlestick_signal,
            'pattern': price_pattern_signal
        },
        'divergences': divergences,
        'divergence_consensus': divergence_consensus,
        'ml_prediction': ml_prediction,
        'convergence_analysis': convergence_analysis,
        'entry_points': entry_points,
        'price_pattern': price_pattern,
        'candlestick_patterns': candlestick_patterns,
        'smc_signals': smc_signals,
        'commodity_signals': commodity_signals,
        'signal_counts': {
            'long': long_count,
            'short': short_count,
            'hold': hold_count,
            'total': total_signals
        },
        'current_price': current_price  # Thêm current_price để sử dụng trong khuyến nghị
    }

def make_decision(analyses):
    """Tổng hợp nhận định từ các khung thời gian
    
    Logic:
    - SIGNAL_THRESHOLD (30%): Ngưỡng tối thiểu để một timeframe được coi là có tín hiệu hợp lệ
    - consensus_ratio: Tỷ lệ đồng thuận thực tế của timeframe có tín hiệu mạnh nhất
    - Chỉ những timeframe có consensus_ratio >= SIGNAL_THRESHOLD mới được xét
    """
    valid_timeframes = []
    
    # Debug logging

    
    for analysis in analyses:
        signal = analysis.get('signal', 'Hold')
        consensus_ratio = analysis.get('consensus_ratio', 0)
        timeframe = analysis.get('timeframe', 'unknown')
        
        if signal in ['Long', 'Short'] and consensus_ratio >= SIGNAL_THRESHOLD:
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
    """Phân tích xu hướng ngắn hạn cho một coin"""
    try:
        logger.info(f"🔍 Bắt đầu phân tích {symbol}...")
        
        # Lấy giá hiện tại cho crypto với fallback
        if exchange:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
        else:
            # Fallback sử dụng yfinance
            symbol_mapping = {
                'BTC/USDT': 'BTC-USD',
                'ETH/USDT': 'ETH-USD'
            }
            
            yf_symbol = symbol_mapping.get(symbol, symbol.replace('/', '-'))
            ticker = yf.Ticker(yf_symbol)
            current_price = ticker.info.get('regularMarketPrice')
            
            if not current_price:
                return None
                
    except Exception as e:
        logger.error(f"Lỗi khi lấy giá hiện tại cho {symbol}: {e}")
        
        # Thử fallback với yfinance
        try:
            symbol_mapping = {
                'BTC/USDT': 'BTC-USD',
                'ETH/USDT': 'ETH-USD'
            }
            
            yf_symbol = symbol_mapping.get(symbol, symbol.replace('/', '-'))
            ticker = yf.Ticker(yf_symbol)
            current_price = ticker.info.get('regularMarketPrice')
            
            if not current_price:
                return None
        except Exception as e2:
            logger.error(f"❌ Fallback cũng thất bại cho {symbol}: {e2}")
            return None

    analyses = []
    for timeframe in TIMEFRAMES:
        # Sử dụng incremental data loading cho phân tích real-time
        if timeframe in ML_TIMEFRAMES:
            # Sử dụng dữ liệu ML đã được cập nhật
            data = load_and_update_historical_data(symbol, timeframe, force_full_update=False)
        else:
            # Sử dụng fetch_ohlcv cho các timeframe không có ML
            data = fetch_ohlcv(symbol, timeframe, CANDLE_LIMIT)
        
        if data is None:
            continue
        
        analysis = analyze_timeframe(data, timeframe, current_price, symbol)
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
            save_prediction(symbol, analysis['timeframe'], prediction_data, current_price)
    
    # Xác minh các dự đoán ML cũ với giá hiện tại
    for timeframe in ML_TIMEFRAMES:
        try:
            verification_result = verify_ml_predictions(symbol, timeframe, current_price, pd.Timestamp.now())
            if verification_result:
                logger.info(f"🔍 Đã xác minh {verification_result['total_checked']} dự đoán ML cho {symbol} ({timeframe})")
        except Exception as e:
            logger.warning(f"⚠️ Không thể xác minh dự đoán ML cho {symbol} ({timeframe}): {e}")

    # Phân tích correlation cho BTC để tăng cường độ chính xác
    if symbol == 'BTC/USDT':
        try:
            # Lấy dữ liệu ETH và BTC Dominance
            eth_data = load_or_fetch_historical_data('ETH/USDT', '4h')
            btc_d_data = get_btc_dominance_data()
            
            if eth_data and btc_d_data:
                # Phân tích correlation
                correlation_analysis = analyze_crypto_correlation_ml(data, eth_data, btc_d_data)
                
                # Tăng cường phân tích BTC với thông tin correlation
                result = enhance_btc_analysis_with_correlation(result, correlation_analysis)
                
                logger.info(f"🔗 Đã phân tích correlation cho BTC: {correlation_analysis.get('analysis', '')}")
            else:
                logger.warning("⚠️ Không thể lấy dữ liệu ETH hoặc BTC Dominance để phân tích correlation")
        except Exception as e:
            logger.error(f"❌ Lỗi phân tích correlation cho BTC: {e}")

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
            logger.info("✅ Đã gửi báo cáo qua Telegram thành công")
            return True
        else:
            logger.error(f"❌ Lỗi khi gửi Telegram: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Lỗi khi gửi Telegram: {e}")
        return False

def format_coin_report(result):
    """Định dạng báo cáo phân tích cho một đồng coin cụ thể - Tối ưu cho tín hiệu mạnh"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    symbol = result['symbol']
    decision = result['decision']
    consensus_ratio = result['consensus_ratio']
    valid_timeframes = result['valid_timeframes']
    
    # Xác định loại tài sản để hiển thị emoji phù hợp
    asset_type = "COIN"
    
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
        report += f"💡 Tín hiệu từ timeframe có đồng thuận cao nhất\n"
        
        # Thêm giải thích lý do cho BTC
        if symbol == 'BTC/USDT':
            report += f"🔍 <b>LÝ DO NHẬN ĐỊNH {symbol}:</b>\n"
            # Phân tích các yếu tố chính
            reasons = []
            
            # Phân tích divergence
            divergence_signals = [a for a in valid_timeframes if a['divergence_consensus']['signal'] != 'Hold']
            if divergence_signals:
                divergence_count = sum(a['divergence_consensus']['count'] for a in divergence_signals)
                reasons.append(f"Divergence mạnh ({divergence_count} signals)")
            
            # Phân tích RSI
            rsi_values = [a['rsi_value'] for a in valid_timeframes]
            if any(rsi < 30 for rsi in rsi_values):
                reasons.append("RSI oversold - cơ hội mua")
            elif any(rsi > 70 for rsi in rsi_values):
                reasons.append("RSI overbought - cảnh báo bán")
            
            # Phân tích MACD
            macd_signals = [a for a in valid_timeframes if a['macd_signal'] != 'Hold']
            if macd_signals:
                reasons.append("MACD crossover - xác nhận xu hướng")
            
            # Phân tích Bollinger Bands
            bb_signals = [a for a in valid_timeframes if a['bb_signal'] != 'Hold']
            if bb_signals:
                reasons.append("Bollinger Bands breakout")
            
            # Phân tích Price Action
            pa_signals = [a for a in valid_timeframes if a.get('price_action_patterns') and len(a['price_action_patterns']) > 0]
            if pa_signals:
                reasons.append("Price Action patterns mạnh")
            
            # Phân tích ML
            ml_signals = [a for a in valid_timeframes if a.get('ml_prediction') and a['ml_prediction']['confidence'] > 0.7]
            if ml_signals:
                ml_avg_conf = sum(a['ml_prediction']['confidence'] for a in ml_signals) / len(ml_signals)
                reasons.append(f"ML prediction cao ({ml_avg_conf:.1%})")
            
            # Hiển thị lý do (tối đa 4 lý do quan trọng nhất)
            if reasons:
                for i, reason in enumerate(reasons[:4], 1):
                    report += f"  {i}. {reason}\n"
            else:
                report += "  • Phân tích kỹ thuật tổng hợp\n"
            
            report += "\n"
        
        # Hiển thị thông tin về các timeframe được chọn
        for analysis in valid_timeframes:
            report += f"📊 <b>{analysis['timeframe']}:</b> {analysis['signal']} (Đồng thuận: {analysis['consensus_ratio']:.1%})\n"
        report += "\n"
        
        # Chỉ hiển thị các tín hiệu mạnh và quan trọng nhất
        for analysis in valid_timeframes:
            timeframe = analysis['timeframe']
            strong_signals = []
            
            # Divergence mạnh - Ưu tiên cao nhất
            if analysis['divergence_consensus']['signal'] != 'Hold' and analysis['divergence_consensus']['strength'] > 0.3:
                strong_signals.append(f"Divergence ({analysis['divergence_consensus']['count']} signals)")
            
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
            
            # Chỉ báo hàng hóa (đã loại bỏ - chỉ phân tích crypto)
            
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
    
    # Thêm thông tin correlation cho BTC
    if symbol == 'BTC/USDT' and 'correlation_analysis' in result:
        correlation = result['correlation_analysis']
        if correlation.get('analysis'):
            report += f"\n🔗 <b>CORRELATION ANALYSIS:</b>\n"
            report += f"📊 BTC-ETH Correlation: {correlation.get('btc_eth_correlation', 0):.2f}\n"
            report += f"📈 BTC Dominance Impact: {correlation.get('btc_dominance_impact', 0):.2f}\n"
            report += f"🎯 Market Sentiment: {correlation.get('market_sentiment', 'neutral').upper()}\n"
            report += f"💡 {correlation.get('analysis', '')}\n"
            
            if correlation.get('ml_signals'):
                report += f"🤖 ML Signals: {', '.join(correlation['ml_signals'])}\n"
    
    return report

def format_analysis_report(results):
    """Định dạng báo cáo phân tích tối ưu với nhấn mạnh divergence/convergence"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Lấy thống kê độ chính xác
    accuracy_data = get_prediction_accuracy_data()
    accuracy_summary = ""
    if accuracy_data:
        overall = accuracy_data.get('overall', {})
        if overall.get('total_predictions', 0) > 0:
            accuracy_summary = f" | 📈 Độ chính xác: {overall['accuracy']:.1%} ({overall['accurate_predictions']}/{overall['total_predictions']})"
    
    report = f"🤖 <b>BÁO CÁO PHÂN TÍCH XU HƯỚNG TỐI ƯU</b>\n"
    report += f"⏰ Thời gian: {current_time}\n"
    report += f"📊 Ngưỡng tối thiểu: {SIGNAL_THRESHOLD:.1%}{accuracy_summary}\n"
    report += f"💰 Tài sản: Crypto\n"
    
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
                report += f"  • {analysis['timeframe']}: {analysis['signal']} ({analysis['confidence']:.1%})\n"
        elif decision in ['Long', 'Short']:
            emoji = "✅" if decision == 'Long' else "🔴"
            report += f"{emoji} <b>{symbol}: {decision}</b> (Độ tin cậy: {consensus_ratio:.1%})\n"
            report += f"📊 Timeframes: {', '.join([a['timeframe'] for a in valid_timeframes])}\n"
            
            # Hiển thị tổng số tín hiệu Long/Short
            total_long = 0
            total_short = 0
            for analysis in valid_timeframes:
                signal_counts = analysis.get('signal_counts', {})
                total_long += signal_counts.get('long', 0)
                total_short += signal_counts.get('short', 0)
            
            report += f"📈 Tổng tín hiệu Long: {total_long} | 📉 Tổng tín hiệu Short: {total_short}\n"
            
            # Thêm thông tin chi tiết cho từng timeframe - chỉ hiển thị top 3
            top_timeframes = sorted(valid_timeframes, key=lambda x: x['consensus_ratio'], reverse=True)[:3]
            for analysis in top_timeframes:
                timeframe = analysis['timeframe']
                report += f"\n📊 <b>{timeframe}:</b>\n"
                
                # === 1. MACHINE LEARNING PREDICTION ===
                ml_prediction = analysis.get('ml_prediction')
                if ml_prediction and ml_prediction.get('confidence', 0) > ML_CONFIDENCE_THRESHOLD:
                    report += f"🤖 <b>ML:</b> {ml_prediction['signal']} ({ml_prediction['model_name']}, {ml_prediction['confidence']:.3f})\n"

                # === 2. CONVERGENCE ANALYSIS ===
                convergence_analysis = analysis.get('convergence_analysis')
                if convergence_analysis and convergence_analysis.get('overall_convergence', 0) > CONVERGENCE_THRESHOLD:
                    report += f"🎯 <b>CONVERGENCE:</b> {convergence_analysis['overall_convergence']:.3f} ({len(convergence_analysis['signals'])} signals)\n"

                # === 3. DIVERGENCE/CONVERGENCE - ƯU TIÊN CAO NHẤT ===
                divergence_consensus = analysis.get('divergence_consensus', {})
                if divergence_consensus.get('signal') != 'Hold' and divergence_consensus.get('strength', 0) > 0.2:
                    strength_emoji = "🔥" if divergence_consensus['strength'] > 0.5 else "⚡"
                    report += f"{strength_emoji} <b>DIVERGENCE:</b> {divergence_consensus['signal']} (Strength: {divergence_consensus['strength']:.2f})\n"
                
                # === 2. CHỈ SỐ CỐT LÕI ===
                signals = analysis.get('signals', {})
                indicators = analysis.get('indicators', {})
                
                # Phân loại các chỉ báo theo tín hiệu
                long_signals = []
                short_signals = []
                hold_signals = []
                
                for signal_name, signal_value in signals.items():
                    if signal_value == 'Long':
                        long_signals.append(signal_name.upper())
                    elif signal_value == 'Short':
                        short_signals.append(signal_name.upper())
                    else:
                        hold_signals.append(signal_name.upper())
                
                # Hiển thị các chỉ báo - Concise
                if long_signals or short_signals:
                    report += f"📊 <b>INDICATORS:</b> "
                    indicators_summary = []
                    
                    # Add key indicator values
                    if 'RSI' in long_signals or 'RSI' in short_signals:
                        indicators_summary.append(f"RSI:{indicators.get('rsi', 0):.0f}")
                    if 'STOCH' in long_signals or 'STOCH' in short_signals:
                        indicators_summary.append(f"Stoch:{indicators.get('stoch_k', 0):.0f}")
                    if 'MACD' in long_signals or 'MACD' in short_signals:
                        indicators_summary.append(f"MACD:{indicators.get('macd_line', 0):.4f}")
                    
                    # Add signal counts
                    if long_signals:
                        indicators_summary.append(f"Long:{len(long_signals)}")
                    if short_signals:
                        indicators_summary.append(f"Short:{len(short_signals)}")
                    
                    report += ' | '.join(indicators_summary) + "\n"
                
                # Hiển thị thông tin về tín hiệu extra và market conditions
                signal_counts = analysis.get('signal_counts', {})
                if signal_counts:
                    total = signal_counts.get('total', 0)
                    if total > 0:
                        long_pct = signal_counts.get('long', 0)/total*100
                        short_pct = signal_counts.get('short', 0)/total*100
                        report += f"📊 <b>SIGNALS:</b> Long {long_pct:.0f}% | Short {short_pct:.0f}% | Total {total}\n"
                
                # Patterns - Concise
                patterns = []
                if analysis.get('price_pattern') != 'None':
                    patterns.append(f"Price: {analysis['price_pattern']}")
                if analysis.get('candlestick_patterns'):
                    patterns.append(f"Candle: {', '.join(analysis['candlestick_patterns'][:2])}")  # Show only first 2
                
                if patterns:
                    report += f"📊 <b>PATTERNS:</b> {' | '.join(patterns)}\n"
                
                # Smart Money Concepts - Concise
                smc_signals = analysis.get('smc_signals', {})
                active_smc = [f"{smc_type}: {smc_signal}" for smc_type, smc_signal in smc_signals.items() if smc_signal != 'Hold']
                if active_smc:
                    report += f"🧠 <b>SMC:</b> {' | '.join(active_smc[:2])}\n"  # Show only first 2
                
                # === SUPPORT/RESISTANCE ANALYSIS ===
                sr_analysis = analysis.get('sr_strength_analysis')
                if sr_analysis:
                    report += f"🎯 <b>S/R:</b> "
                    
                    # Support Analysis - Ultra concise
                    if sr_analysis.get('support_analysis') and sr_analysis['support_analysis'].get('price'):
                        support_info = sr_analysis['support_analysis']
                        price = support_info.get('price', 0)
                        if price > 0:
                            report += f"📈${price:.0f} "
                        else:
                            logger.warning(f"⚠️ Support price is 0 or invalid: {support_info}")
                    else:
                        logger.warning(f"⚠️ No support_analysis or price found in sr_analysis: {sr_analysis}")
                    
                    # Resistance Analysis - Ultra concise
                    if sr_analysis.get('resistance_analysis') and sr_analysis['resistance_analysis'].get('price'):
                        resistance_info = sr_analysis['resistance_analysis']
                        price = resistance_info.get('price', 0)
                        if price > 0:
                            report += f"📉${price:.0f} "
                        else:
                            logger.warning(f"⚠️ Resistance price is 0 or invalid: {resistance_info}")
                    else:
                        logger.warning(f"⚠️ No resistance_analysis or price found in sr_analysis: {sr_analysis}")
                    
                    # Breakout Potential - Ultra concise
                    if sr_analysis.get('breakout_potential'):
                        breakout_info = sr_analysis['breakout_potential']
                        report += f"🚀{breakout_info.get('direction', 'Unknown')[:3]} "
                    
                    report += "\n"
                
                # Support/Resistance Details - Ultra concise
                sr_details = analysis.get('support_resistance_analysis')
                if sr_details:
                    report += f"🔍 <b>LEVELS:</b> "
                    
                    # Top Support Levels - Show only top 1
                    if sr_details.get('all_support_levels'):
                        level = sr_details['all_support_levels'][0]
                        report += f"📈${level[1]:.0f} "
                    
                    # Top Resistance Levels - Show only top 1
                    if sr_details.get('all_resistance_levels'):
                        level = sr_details['all_resistance_levels'][0]
                        report += f"📉${level[1]:.0f} "
                    
                    # Key Fibonacci Levels only - show only 38.2% and 61.8%
                    if sr_details.get('fibonacci_levels'):
                        fib_levels = sr_details['fibonacci_levels']
                        if '38.2%' in fib_levels:
                            report += f"📐38.2%:${fib_levels['38.2%']:.0f} "
                        if '61.8%' in fib_levels:
                            report += f"61.8%:${fib_levels['61.8%']:.0f} "
                    
                                        # Thêm khuyến nghị giao dịch dựa trên LEVELS
                    current_price = analysis.get('current_price', 0)
                    if current_price > 0:
                        report += "\n💡 <b>KHUYẾN NGHỊ:</b> "
                        
                        # Lấy các mức quan trọng
                        support_levels = sr_details.get('all_support_levels', [])
                        resistance_levels = sr_details.get('all_resistance_levels', [])
                        
                        if support_levels and resistance_levels:
                            nearest_support = support_levels[0][1]  # Giá support gần nhất
                            nearest_resistance = resistance_levels[0][1]  # Giá resistance gần nhất
                            
                            # Tạo khuyến nghị từ các mức support/resistance
                            recommendation = generate_trading_recommendation(
                                current_price, nearest_support, nearest_resistance
                            )
                            report += recommendation
                    
                    report += "\n"
                
                # Fallback: Nếu không có S/R từ sr_strength_analysis, hiển thị từ support_resistance_analysis
                elif not sr_analysis and sr_details:
                    report += f"🎯 <b>S/R:</b> "
                    
                    # Hiển thị support/resistance cơ bản
                    if sr_details.get('nearest_support') and sr_details['nearest_support'].get('price'):
                        support_price = sr_details['nearest_support']['price']
                        if support_price > 0:
                            report += f"📈${support_price:.0f} "
                    
                    if sr_details.get('nearest_resistance') and sr_details['nearest_resistance'].get('price'):
                        resistance_price = sr_details['nearest_resistance']['price']
                        if resistance_price > 0:
                            report += f"📉${resistance_price:.0f} "
                    
                    # Thêm khuyến nghị giao dịch cho fallback
                    current_price = analysis.get('current_price', 0)
                    if current_price > 0:
                        report += "\n💡 <b>KHUYẾN NGHỊ:</b> "
                        
                        if support_price > 0 and resistance_price > 0:
                            # Tạo khuyến nghị từ các mức support/resistance
                            recommendation = generate_trading_recommendation(
                                current_price, support_price, resistance_price
                            )
                            report += recommendation
                    
                    report += "\n"
                
                # Commodity Signals (đã loại bỏ - chỉ phân tích crypto)
                

                
                # Entry Points - Ultra concise
                if 'entry_points' in analysis:
                    entry = analysis['entry_points']
                    report += f"🎯 <b>ENTRY:</b> ${entry['aggressive']:.0f} | SL: ${entry['stop_loss']:.0f} | TP: ${entry['take_profit']:.0f}\n"
                
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
                emoji = "🟢"
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
                # Chỉ gửi báo cáo BTC, nhưng vẫn phân tích cả BTC và ETH
                for symbol in SYMBOLS:
                    result = analyze_coin(symbol)
                    if result:
                        results.append(result)
                
                # Gửi báo cáo chỉ cho BTC
                if results:
                    logger.info(f"🔍 Lọc {len(results)} kết quả để chỉ gửi BTC...")
                    for result in results:
                        if result['symbol'] == 'BTC/USDT':  # Chỉ gửi báo cáo BTC
                            try:
                                coin_report = format_coin_report(result)
                                send_telegram_message(coin_report)
                                logger.info(f"✅ Đã gửi báo cáo cho {result['symbol']}")
                            except Exception as e:
                                logger.error(f"❌ Lỗi khi gửi báo cáo cho {result['symbol']}: {e}")
                        else:
                            logger.info(f"📊 Đã phân tích {result['symbol']} nhưng KHÔNG GỬI BÁO CÁO (theo yêu cầu)")
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

def ml_model_trainer_scheduler():
    """Lập lịch train ML models định kỳ"""
    def train_models_periodically():
        while True:
            try:
                logger.info("🤖 Bắt đầu train ML models cho BTC và ETH...")
                
                # Chỉ train cho BTC và ETH theo yêu cầu
                ml_symbols = ['BTC/USDT', 'ETH/USDT']
                ml_timeframes = ['1h', '4h', '1d']  # Chỉ train cho 3 timeframe chính
                
                for symbol in ml_symbols:
                    for timeframe in ml_timeframes:
                        try:
                            logger.info(f"🔄 Training ML models cho {symbol} ({timeframe})...")
                            
                            # Kiểm tra dữ liệu trước khi train
                            data = load_or_fetch_historical_data(symbol, timeframe)
                            if data is None:
                                logger.warning(f"⚠️ Không thể lấy dữ liệu cho {symbol} ({timeframe})")
                                continue
                            
                            logger.info(f"📊 Dữ liệu {symbol} ({timeframe}): {len(data['close'])} candles")
                            
                            trained_models = train_ml_models(symbol, timeframe)
                            if trained_models:
                                logger.info(f"✅ Đã train thành công {len(trained_models)} models cho {symbol} ({timeframe})")
                            else:
                                logger.warning(f"⚠️ Không thể train models cho {symbol} ({timeframe})")
                            
                            time.sleep(5)  # Chờ 5 giây giữa các lần train
                        except Exception as e:
                            logger.error(f"❌ Lỗi khi train ML cho {symbol} ({timeframe}): {e}")
                            continue
                
                logger.info(f"⏰ Chờ {ML_UPDATE_INTERVAL} giây để train ML models tiếp theo...")
                time.sleep(ML_UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Lỗi trong ml_model_trainer_scheduler: {e}")
                time.sleep(3600)  # Chờ 1 giờ nếu có lỗi
    
    # Khởi động thread train ML models định kỳ
    ml_thread = threading.Thread(target=train_models_periodically, daemon=True)
    ml_thread.start()
    logger.info(f"🤖 Đã khởi động ML Model Trainer Scheduler (train mỗi {ML_UPDATE_INTERVAL//3600} giờ)")

def calculate_entry_points(current_price, highs, lows, closes, rsi, bb_upper, bb_lower, ema50, pivot_points, support, resistance, support_resistance_analysis=None, signal='Hold'):
    """Tính toán các điểm entry hợp lý với phân tích support/resistance nâng cao và khuyến nghị từ chỉ báo"""
    
    # === CHỈ TÍNH ENTRY KHI CÓ TÍN HIỆU RÕ RÀNG ===
    if signal not in ['Long', 'Short']:
        # Trả về entry points mặc định khi không có tín hiệu rõ ràng
        return {
            'immediate': current_price,
            'conservative': current_price,
            'aggressive': current_price,
            'stop_loss': 0,  # Không có SL khi không có tín hiệu
            'take_profit': 0,  # Không có TP khi không có tín hiệu
            'analysis': ["⚠️ KHÔNG CÓ TÍN HIỆU RÕ RÀNG - Không đưa ra entry points để tránh mâu thuẫn"]
        }
    
    entry_points = {
        'immediate': current_price,
        'conservative': current_price,
        'aggressive': current_price,
        'stop_loss': current_price,
        'take_profit': current_price,
        'analysis': []
    }
    
    # Helper function để lấy giá trị cuối cùng
    def get_last(series):
        return series.iloc[-1] if hasattr(series, 'iloc') else series[-1]
    
    # Helper function để tính ATR fallback
    def calculate_atr_fallback():
        recent_ranges = []
        for i in range(max(0, len(highs)-10), len(highs)):
            if hasattr(highs, 'iloc'):
                recent_ranges.append(highs.iloc[i] - lows.iloc[i])
            else:
                recent_ranges.append(highs[i] - lows[i])
        return np.mean(recent_ranges) if recent_ranges else (current_price * 0.02)
    
    # 1. Sử dụng khuyến nghị từ chỉ báo thay vì chỉ dựa vào trend
    if signal == 'Long':
        trend = 'bullish'
    elif signal == 'Short':
        trend = 'bearish'
    else:
        # Fallback: Phân tích xu hướng hiện tại nếu không có khuyến nghị
        if current_price > get_last(ema50):
            trend = 'bullish'
        else:
            trend = 'bearish'
    
    # 2. Sử dụng thông tin support/resistance nâng cao nếu có
    if support_resistance_analysis:
        # Lấy các mức support/resistance mạnh nhất
        strong_support = None
        strong_resistance = None
        
        # Tìm support mạnh nhất
        if support_resistance_analysis.get('all_support_levels'):
            for level in support_resistance_analysis['all_support_levels']:
                if level[3] > 0.7:  # Strength > 0.7
                    strong_support = level[1]
                    break
        
        # Tìm resistance mạnh nhất
        if support_resistance_analysis.get('all_resistance_levels'):
            for level in support_resistance_analysis['all_resistance_levels']:
                if level[3] > 0.7:  # Strength > 0.7
                    strong_resistance = level[1]
                    break
        
        # Cập nhật support/resistance nếu tìm được mức mạnh hơn
        if strong_support and strong_support < current_price:
            support = strong_support
        if strong_resistance and strong_resistance > current_price:
            resistance = strong_resistance
    
    # 3. Tính các mức entry cho Long
    if trend == 'bullish':
        # Entry bảo thủ (Conservative) - Chờ pullback về hỗ trợ
        conservative_entry = min(support, get_last(bb_lower), pivot_points['s1'])
        entry_points['conservative'] = conservative_entry
        
        # Entry tích cực (Aggressive) - Vào ngay khi có tín hiệu
        aggressive_entry = current_price * 0.995  # Vào thấp hơn giá hiện tại 0.5%
        entry_points['aggressive'] = aggressive_entry
        
        # Stop Loss - Dựa trên mức hỗ trợ mạnh (s2) để tạo R/R tốt hơn
        # Sử dụng s2 thay vì s1 để SL gần entry hơn
        stop_loss = min(support * 0.998, get_last(bb_lower) * 0.999, pivot_points['s2'] * 0.999)
        
        # === VALIDATION: Đảm bảo SL luôn thấp hơn entry ===
        if stop_loss >= aggressive_entry:
            # Fallback: Sử dụng ATR để tính SL hợp lý
            atr = calculate_atr_fallback()
            stop_loss = aggressive_entry - (atr * 0.5)  # SL = Entry - 0.5*ATR
            entry_points['analysis'].append(f"  • SL được điều chỉnh bằng ATR để đảm bảo logic")
        
        entry_points['stop_loss'] = stop_loss
        
        # Take Profit - Tỷ lệ với khoảng cách SL để tạo R/R ít nhất 1:2
        sl_distance = aggressive_entry - stop_loss
        if sl_distance > 0:
            # TP = Entry + (SL_distance * 2.5) để có R/R ít nhất 1:2.5
            take_profit = aggressive_entry + (sl_distance * 2.5)
        else:
            # Fallback nếu không tính được SL distance
            atr = calculate_atr_fallback()
            take_profit = aggressive_entry + (atr * 1.5)
        entry_points['take_profit'] = take_profit
        
        entry_points['analysis'].append(f"📈 XU HƯỚNG TĂNG - Điểm entry hợp lý:")
        entry_points['analysis'].append(f"  • Entry bảo thủ: ${conservative_entry:.4f} (chờ pullback)")
        entry_points['analysis'].append(f"  • Entry tích cực: ${aggressive_entry:.4f} (vào ngay)")
        entry_points['analysis'].append(f"  • Stop Loss: ${stop_loss:.4f}")
        entry_points['analysis'].append(f"  • Take Profit: ${take_profit:.4f}")
    
    # 4. Tính các mức entry cho Short
    elif trend == 'bearish':
        # Entry bảo thủ - Chờ bounce về kháng cự
        conservative_entry = max(resistance, get_last(bb_upper), pivot_points['r1'])
        entry_points['conservative'] = conservative_entry
        
        # Entry tích cực - Vào ngay khi có tín hiệu
        aggressive_entry = current_price * 1.005  # Vào cao hơn giá hiện tại 0.5%
        entry_points['aggressive'] = aggressive_entry
        
        # Stop Loss - Dựa trên mức kháng cự mạnh (r2) để tạo R/R tốt hơn
        # Sử dụng r2 thay vì r1 để SL gần entry hơn
        stop_loss = max(resistance * 1.002, get_last(bb_upper) * 1.001, pivot_points['r2'] * 1.001)
        
        # === VALIDATION: Đảm bảo SL luôn cao hơn entry ===
        if stop_loss <= aggressive_entry:
            # Fallback: Sử dụng ATR để tính SL hợp lý
            atr = calculate_atr_fallback()
            stop_loss = aggressive_entry + (atr * 0.5)  # SL = Entry + 0.5*ATR
            entry_points['analysis'].append(f"  • SL được điều chỉnh bằng ATR để đảm bảo logic")
        
        entry_points['stop_loss'] = stop_loss
        
        # Take Profit - Tỷ lệ với khoảng cách SL để tạo R/R ít nhất 1:2
        sl_distance = stop_loss - aggressive_entry
        if sl_distance > 0:
            # TP = Entry - (SL_distance * 2.5) để có R/R ít nhất 1:2.5
            take_profit = aggressive_entry - (sl_distance * 2.5)
        else:
            # Fallback nếu không tính được SL distance
            atr = calculate_atr_fallback()
            take_profit = aggressive_entry - (atr * 1.5)
        entry_points['take_profit'] = take_profit
        
        entry_points['analysis'].append(f"📉 XU HƯỚNG GIẢM - Điểm entry hợp lý:")
        entry_points['analysis'].append(f"  • Entry bảo thủ: ${conservative_entry:.4f} (chờ bounce)")
        entry_points['analysis'].append(f"  • Entry tích cực: ${aggressive_entry:.4f} (vào ngay)")
        entry_points['analysis'].append(f"  • Stop Loss: ${stop_loss:.4f}")
        entry_points['analysis'].append(f"  • Take Profit: ${take_profit:.4f}")
    
    # 5. Phân tích RSI để tối ưu entry
    rsi_value = get_last(rsi)
    if rsi_value < 15:  # Từ 20 -> 15
        entry_points['analysis'].append(f"  • RSI quá bán ({rsi_value:.1f}) → Ưu tiên entry bảo thủ")
    elif rsi_value > 85:  # Từ 80 -> 85
        entry_points['analysis'].append(f"  • RSI quá mua ({rsi_value:.1f}) → Ưu tiên entry bảo thủ")
    else:
        entry_points['analysis'].append(f"  • RSI trung tính ({rsi_value:.1f}) → Có thể entry tích cực")
    
    # 6. Phân tích Bollinger Bands
    if current_price < get_last(bb_lower):
        entry_points['analysis'].append(f"  • Giá dưới BB Lower → Cơ hội entry tốt cho Long")
    elif current_price > get_last(bb_upper):
        entry_points['analysis'].append(f"  • Giá trên BB Upper → Cơ hội entry tốt cho Short")
    else:
        entry_points['analysis'].append(f"  • Giá trong BB → Entry ở giữa range")
    
    # 7. Phân tích Support/Resistance Strength
    if support_resistance_analysis:
        sr_analysis = support_resistance_analysis.get('sr_strength_analysis')
        if sr_analysis:
            # Support Strength Analysis
            if sr_analysis.get('support_analysis'):
                support_strength = sr_analysis['support_analysis'].get('strength', 0)
                if support_strength > 0.8:
                    entry_points['analysis'].append(f"  • Support mạnh (Strength: {support_strength:.2f}) → Entry bảo thủ an toàn")
                elif support_strength < 0.4:
                    entry_points['analysis'].append(f"  • Support yếu (Strength: {support_strength:.2f}) → Cẩn thận với breakdown")
            
            # Resistance Strength Analysis
            if sr_analysis.get('resistance_analysis'):
                resistance_strength = sr_analysis['resistance_analysis'].get('strength', 0)
                if resistance_strength > 0.8:
                    entry_points['analysis'].append(f"  • Resistance mạnh (Strength: {resistance_strength:.2f}) → Khó breakout")
                elif resistance_strength < 0.4:
                    entry_points['analysis'].append(f"  • Resistance yếu (Strength: {resistance_strength:.2f}) → Dễ breakout")
            
            # Breakout Potential
            if sr_analysis.get('breakout_potential'):
                breakout_direction = sr_analysis['breakout_potential'].get('direction', 'Unknown')
                breakout_probability = sr_analysis['breakout_potential'].get('probability', 'Unknown')
                entry_points['analysis'].append(f"  • Breakout tiềm năng: {breakout_direction} (Xác suất: {breakout_probability})")
    
    # 8. Tính Risk/Reward Ratio
    if trend == 'bullish':
        risk = entry_points['aggressive'] - entry_points['stop_loss']
        reward = entry_points['take_profit'] - entry_points['aggressive']
        rr_ratio = reward / risk if risk > 0 else 0
        entry_points['analysis'].append(f"  • Risk/Reward Ratio: 1:{rr_ratio:.2f}")
    elif trend == 'bearish':
        risk = entry_points['stop_loss'] - entry_points['aggressive']
        reward = entry_points['aggressive'] - entry_points['take_profit']
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

def detect_divergence(price_data, indicator_data, lookback=14):
    """
    Phát hiện divergence giữa giá và chỉ báo
    Returns: {'type': 'bullish/bearish/hidden_bullish/hidden_bearish', 'strength': 0-1}
    """
    if len(price_data) < lookback * 2:
        return None
    
    # Chuyển đổi sang pandas Series nếu cần
    if not isinstance(price_data, pd.Series):
        price_data = pd.Series(price_data)
    if not isinstance(indicator_data, pd.Series):
        indicator_data = pd.Series(indicator_data)
    
    # Tìm các đỉnh và đáy trong giá
    price_peaks = []
    price_troughs = []
    
    for i in range(1, len(price_data) - 1):
        if price_data.iloc[i] > price_data.iloc[i-1] and price_data.iloc[i] > price_data.iloc[i+1]:
            price_peaks.append((i, price_data.iloc[i]))
        elif price_data.iloc[i] < price_data.iloc[i-1] and price_data.iloc[i] < price_data.iloc[i+1]:
            price_troughs.append((i, price_data.iloc[i]))
    
    # Tìm các đỉnh và đáy trong chỉ báo
    indicator_peaks = []
    indicator_troughs = []
    
    for i in range(1, len(indicator_data) - 1):
        if indicator_data.iloc[i] > indicator_data.iloc[i-1] and indicator_data.iloc[i] > indicator_data.iloc[i+1]:
            indicator_peaks.append((i, indicator_data.iloc[i]))
        elif indicator_data.iloc[i] < indicator_data.iloc[i-1] and indicator_data.iloc[i] < indicator_data.iloc[i+1]:
            indicator_troughs.append((i, indicator_data.iloc[i]))
    
    # Phân tích divergence
    divergence_result = None
    
    # Regular Bullish Divergence: Giá tạo đáy thấp hơn, chỉ báo tạo đáy cao hơn
    if len(price_troughs) >= 2 and len(indicator_troughs) >= 2:
        recent_price_trough = price_troughs[-1]
        prev_price_trough = price_troughs[-2]
        recent_indicator_trough = indicator_troughs[-1]
        prev_indicator_trough = indicator_troughs[-2]
        
        if (recent_price_trough[1] < prev_price_trough[1] and 
            recent_indicator_trough[1] > prev_indicator_trough[1] and
            recent_price_trough[0] > prev_price_trough[0] and
            recent_indicator_trough[0] > prev_indicator_trough[0]):
            
            strength = min(1.0, abs(recent_price_trough[1] - prev_price_trough[1]) / prev_price_trough[1])
            divergence_result = {'type': 'bullish', 'strength': strength}
    
    # Regular Bearish Divergence: Giá tạo đỉnh cao hơn, chỉ báo tạo đỉnh thấp hơn
    elif len(price_peaks) >= 2 and len(indicator_peaks) >= 2:
        recent_price_peak = price_peaks[-1]
        prev_price_peak = price_peaks[-2]
        recent_indicator_peak = indicator_peaks[-1]
        prev_indicator_peak = indicator_peaks[-2]
        
        if (recent_price_peak[1] > prev_price_peak[1] and 
            recent_indicator_peak[1] < prev_indicator_peak[1] and
            recent_price_peak[0] > prev_price_peak[0] and
            recent_indicator_peak[0] > prev_indicator_peak[0]):
            
            strength = min(1.0, abs(recent_price_peak[1] - prev_price_peak[1]) / prev_price_peak[1])
            divergence_result = {'type': 'bearish', 'strength': strength}
    
    # Hidden Bullish Divergence: Giá tạo đáy cao hơn, chỉ báo tạo đáy thấp hơn
    elif len(price_troughs) >= 2 and len(indicator_troughs) >= 2:
        recent_price_trough = price_troughs[-1]
        prev_price_trough = price_troughs[-2]
        recent_indicator_trough = indicator_troughs[-1]
        prev_indicator_trough = indicator_troughs[-2]
        
        if (recent_price_trough[1] > prev_price_trough[1] and 
            recent_indicator_trough[1] < prev_indicator_trough[1] and
            recent_price_trough[0] > prev_price_trough[0] and
            recent_indicator_trough[0] > prev_indicator_trough[0]):
            
            strength = min(1.0, abs(recent_price_trough[1] - prev_price_trough[1]) / prev_price_trough[1])
            divergence_result = {'type': 'hidden_bullish', 'strength': strength}
    
    # Hidden Bearish Divergence: Giá tạo đỉnh thấp hơn, chỉ báo tạo đỉnh cao hơn
    elif len(price_peaks) >= 2 and len(indicator_peaks) >= 2:
        recent_price_peak = price_peaks[-1]
        prev_price_peak = price_peaks[-2]
        recent_indicator_peak = indicator_peaks[-1]
        prev_indicator_peak = indicator_peaks[-2]
        
        if (recent_price_peak[1] < prev_price_peak[1] and 
            recent_indicator_peak[1] > prev_indicator_peak[1] and
            recent_price_peak[0] > prev_price_peak[0] and
            recent_indicator_peak[0] > prev_indicator_peak[0]):
            
            strength = min(1.0, abs(recent_price_peak[1] - prev_price_peak[1]) / prev_price_peak[1])
            divergence_result = {'type': 'hidden_bearish', 'strength': strength}
    
    return divergence_result

def analyze_rsi_divergence(close_prices, rsi_values):
    """Phân tích RSI divergence"""
    if len(close_prices) < 20 or len(rsi_values) < 20:
        return None
    
    # Lọc dữ liệu không null
    valid_indices = []
    for i in range(len(close_prices)):
        if not np.isnan(close_prices[i]) and not np.isnan(rsi_values[i]):
            valid_indices.append(i)
    
    if len(valid_indices) < 20:
        return None
    
    close_clean = [close_prices[i] for i in valid_indices]
    rsi_clean = [rsi_values[i] for i in valid_indices]
    
    divergence = detect_divergence(close_clean, rsi_clean, lookback=14)
    
    if divergence:
        # Thêm thông tin chi tiết
        divergence['indicator'] = 'RSI'
        divergence['description'] = get_divergence_description(divergence['type'], 'RSI')
        divergence['signal'] = get_divergence_signal(divergence['type'])
    
    return divergence

def analyze_macd_divergence(close_prices, macd_values):
    """Phân tích MACD divergence"""
    if len(close_prices) < 20 or len(macd_values) < 20:
        return None
    
    # Lọc dữ liệu không null
    valid_indices = []
    for i in range(len(close_prices)):
        if not np.isnan(close_prices[i]) and not np.isnan(macd_values[i]):
            valid_indices.append(i)
    
    if len(valid_indices) < 20:
        return None
    
    close_clean = [close_prices[i] for i in valid_indices]
    macd_clean = [macd_values[i] for i in valid_indices]
    
    divergence = detect_divergence(close_clean, macd_clean, lookback=14)
    
    if divergence:
        # Thêm thông tin chi tiết
        divergence['indicator'] = 'MACD'
        divergence['description'] = get_divergence_description(divergence['type'], 'MACD')
        divergence['signal'] = get_divergence_signal(divergence['type'])
    
    return divergence

def analyze_price_volume_divergence(close_prices, volume_data):
    """Phân tích divergence giữa giá và khối lượng"""
    if len(close_prices) < 20 or len(volume_data) < 20:
        return None
    
    # Tính toán volume moving average để so sánh
    volume_ma = ta.trend.sma_indicator(volume_data, window=10)
    
    # Lọc dữ liệu không null
    valid_indices = []
    for i in range(len(close_prices)):
        if not np.isnan(close_prices[i]) and not np.isnan(volume_ma[i]):
            valid_indices.append(i)
    
    if len(valid_indices) < 20:
        return None
    
    close_clean = [close_prices[i] for i in valid_indices]
    volume_ma_clean = [volume_ma[i] for i in valid_indices]
    
    divergence = detect_divergence(close_clean, volume_ma_clean, lookback=14)
    
    if divergence:
        # Thêm thông tin chi tiết
        divergence['indicator'] = 'Volume'
        divergence['description'] = get_divergence_description(divergence['type'], 'Volume')
        divergence['signal'] = get_divergence_signal(divergence['type'])
    
    return divergence

def get_divergence_description(divergence_type, indicator):
    """Tạo mô tả chi tiết cho divergence"""
    descriptions = {
        'bullish': {
            'RSI': 'RSI tạo đáy cao hơn trong khi giá tạo đáy thấp hơn → Tín hiệu đảo chiều tăng mạnh',
            'MACD': 'MACD tạo đáy cao hơn trong khi giá tạo đáy thấp hơn → Tín hiệu đảo chiều tăng mạnh',
            'Volume': 'Khối lượng tăng trong khi giá giảm → Tín hiệu tích lũy, sẵn sàng đảo chiều tăng'
        },
        'bearish': {
            'RSI': 'RSI tạo đỉnh thấp hơn trong khi giá tạo đỉnh cao hơn → Tín hiệu đảo chiều giảm mạnh',
            'MACD': 'MACD tạo đỉnh thấp hơn trong khi giá tạo đỉnh cao hơn → Tín hiệu đảo chiều giảm mạnh',
            'Volume': 'Khối lượng giảm trong khi giá tăng → Tín hiệu phân phối, sẵn sàng đảo chiều giảm'
        },
        'hidden_bullish': {
            'RSI': 'RSI tạo đáy thấp hơn trong khi giá tạo đáy cao hơn → Xu hướng tăng tiếp diễn',
            'MACD': 'MACD tạo đáy thấp hơn trong khi giá tạo đáy cao hơn → Xu hướng tăng tiếp diễn',
            'Volume': 'Khối lượng giảm trong khi giá tăng → Xu hướng tăng tiếp diễn'
        },
        'hidden_bearish': {
            'RSI': 'RSI tạo đỉnh cao hơn trong khi giá tạo đỉnh thấp hơn → Xu hướng giảm tiếp diễn',
            'MACD': 'MACD tạo đỉnh cao hơn trong khi giá tạo đỉnh thấp hơn → Xu hướng giảm tiếp diễn',
            'Volume': 'Khối lượng tăng trong khi giá giảm → Xu hướng giảm tiếp diễn'
        }
    }
    
    return descriptions.get(divergence_type, {}).get(indicator, f'{divergence_type} divergence detected')

def get_divergence_signal(divergence_type):
    """Chuyển đổi loại divergence thành tín hiệu giao dịch"""
    signal_map = {
        'bullish': 'Long',
        'hidden_bullish': 'Long',
        'bearish': 'Short',
        'hidden_bearish': 'Short'
    }
    return signal_map.get(divergence_type, 'Hold')

def analyze_all_divergences(close_prices, rsi_values, macd_values, volume_data):
    """Phân tích tất cả các loại divergence theo phương pháp trading Việt Nam"""
    divergences = []
    
    # RSI Divergence (ưu tiên cao cho H4)
    rsi_div = analyze_rsi_divergence_enhanced(close_prices, rsi_values)
    if rsi_div:
        divergences.append(rsi_div)
    
    # MACD Divergence (ưu tiên cao cho H4)
    macd_div = analyze_macd_divergence_enhanced(close_prices, macd_values)
    if macd_div:
        divergences.append(macd_div)
    
    # Price-Volume Divergence
    volume_div = analyze_price_volume_divergence(close_prices, volume_data)
    if volume_div:
        divergences.append(volume_div)
    
    # Hidden Divergence (theo Elliott Wave)
    hidden_div = analyze_hidden_divergence(close_prices, rsi_values, macd_values)
    if hidden_div:
        divergences.append(hidden_div)
    
    return divergences

def analyze_rsi_divergence_enhanced(close_prices, rsi_values):
    """Phân tích RSI divergence nâng cao cho H4 timeframe"""
    try:
        if len(close_prices) < 20:
            return None
        
        # Lấy 20 giá trị gần nhất
        recent_closes = close_prices.iloc[-20:] if hasattr(close_prices, 'iloc') else close_prices[-20:]
        recent_rsi = rsi_values.iloc[-20:] if hasattr(rsi_values, 'iloc') else rsi_values[-20:]
        
        # Tìm swing highs và swing lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent_closes) - 2):
            if (recent_closes.iloc[i] if hasattr(recent_closes, 'iloc') else recent_closes[i]) > \
               max(recent_closes.iloc[i-2:i+3] if hasattr(recent_closes, 'iloc') else recent_closes[i-2:i+3]):
                swing_highs.append((i, recent_closes.iloc[i] if hasattr(recent_closes, 'iloc') else recent_closes[i], 
                                  recent_rsi.iloc[i] if hasattr(recent_rsi, 'iloc') else recent_rsi[i]))
            
            if (recent_closes.iloc[i] if hasattr(recent_closes, 'iloc') else recent_closes[i]) < \
               min(recent_closes.iloc[i-2:i+3] if hasattr(recent_closes, 'iloc') else recent_closes[i-2:i+3]):
                swing_lows.append((i, recent_closes.iloc[i] if hasattr(recent_closes, 'iloc') else recent_closes[i], 
                                 recent_rsi.iloc[i] if hasattr(recent_rsi, 'iloc') else recent_rsi[i]))
        
        # Kiểm tra divergence
        if len(swing_highs) >= 2:
            # Bearish Divergence
            last_high = swing_highs[-1]
            prev_high = swing_highs[-2]
            
            if last_high[1] > prev_high[1] and last_high[2] < prev_high[2]:
                return {
                    'type': 'bearish_divergence',
                    'indicator': 'RSI',
                    'strength': 0.8,
                    'signal': 'Short',
                    'analysis': f'RSI Bearish Divergence: Giá cao hơn nhưng RSI thấp hơn ({last_high[2]:.1f} vs {prev_high[2]:.1f})'
                }
        
        if len(swing_lows) >= 2:
            # Bullish Divergence
            last_low = swing_lows[-1]
            prev_low = swing_lows[-2]
            
            if last_low[1] < prev_low[1] and last_low[2] > prev_low[2]:
                return {
                    'type': 'bullish_divergence',
                    'indicator': 'RSI',
                    'strength': 0.8,
                    'signal': 'Long',
                    'analysis': f'RSI Bullish Divergence: Giá thấp hơn nhưng RSI cao hơn ({last_low[2]:.1f} vs {prev_low[2]:.1f})'
                }
        
        return None
    except Exception as e:
        logger.error(f"Lỗi phân tích RSI divergence nâng cao: {e}")
        return None

def analyze_macd_divergence_enhanced(close_prices, macd_values):
    """Phân tích MACD divergence nâng cao cho H4 timeframe"""
    try:
        if len(close_prices) < 20:
            return None
        
        # Lấy 20 giá trị gần nhất
        recent_closes = close_prices.iloc[-20:] if hasattr(close_prices, 'iloc') else close_prices[-20:]
        recent_macd = macd_values.iloc[-20:] if hasattr(macd_values, 'iloc') else macd_values[-20:]
        
        # Tìm swing highs và swing lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent_closes) - 2):
            if (recent_closes.iloc[i] if hasattr(recent_closes, 'iloc') else recent_closes[i]) > \
               max(recent_closes.iloc[i-2:i+3] if hasattr(recent_closes, 'iloc') else recent_closes[i-2:i+3]):
                swing_highs.append((i, recent_closes.iloc[i] if hasattr(recent_closes, 'iloc') else recent_closes[i], 
                                  recent_macd.iloc[i] if hasattr(recent_macd, 'iloc') else recent_macd[i]))
            
            if (recent_closes.iloc[i] if hasattr(recent_closes, 'iloc') else recent_closes[i]) < \
               min(recent_closes.iloc[i-2:i+3] if hasattr(recent_closes, 'iloc') else recent_closes[i-2:i+3]):
                swing_lows.append((i, recent_closes.iloc[i] if hasattr(recent_closes, 'iloc') else recent_closes[i], 
                                 recent_macd.iloc[i] if hasattr(recent_macd, 'iloc') else recent_macd[i]))
        
        # Kiểm tra divergence
        if len(swing_highs) >= 2:
            # Bearish Divergence
            last_high = swing_highs[-1]
            prev_high = swing_highs[-2]
            
            if last_high[1] > prev_high[1] and last_high[2] < prev_high[2]:
                return {
                    'type': 'bearish_divergence',
                    'indicator': 'MACD',
                    'strength': 0.8,
                    'signal': 'Short',
                    'analysis': f'MACD Bearish Divergence: Giá cao hơn nhưng MACD thấp hơn ({last_high[2]:.4f} vs {prev_high[2]:.4f})'
                }
        
        if len(swing_lows) >= 2:
            # Bullish Divergence
            last_low = swing_lows[-1]
            prev_low = swing_lows[-2]
            
            if last_low[1] < prev_low[1] and last_low[2] > prev_low[2]:
                return {
                    'type': 'bullish_divergence',
                    'indicator': 'MACD',
                    'strength': 0.8,
                    'signal': 'Long',
                    'analysis': f'MACD Bullish Divergence: Giá thấp hơn nhưng MACD cao hơn ({last_low[2]:.4f} vs {prev_low[2]:.4f})'
                }
        
        return None
    except Exception as e:
        logger.error(f"Lỗi phân tích MACD divergence nâng cao: {e}")
        return None

def analyze_hidden_divergence(close_prices, rsi_values, macd_values):
    """Phân tích Hidden Divergence theo Elliott Wave theory"""
    try:
        if len(close_prices) < 15:
            return None
        
        # Lấy 15 giá trị gần nhất
        recent_closes = close_prices.iloc[-15:] if hasattr(close_prices, 'iloc') else close_prices[-15:]
        recent_rsi = rsi_values.iloc[-15:] if hasattr(rsi_values, 'iloc') else rsi_values[-15:]
        recent_macd = macd_values.iloc[-15:] if hasattr(macd_values, 'iloc') else macd_values[-15:]
        
        # Tìm swing highs và swing lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent_closes) - 2):
            if (recent_closes.iloc[i] if hasattr(recent_closes, 'iloc') else recent_closes[i]) > \
               max(recent_closes.iloc[i-2:i+3] if hasattr(recent_closes, 'iloc') else recent_closes[i-2:i+3]):
                swing_highs.append((i, recent_closes.iloc[i] if hasattr(recent_closes, 'iloc') else recent_closes[i], 
                                  recent_rsi.iloc[i] if hasattr(recent_rsi, 'iloc') else recent_rsi[i],
                                  recent_macd.iloc[i] if hasattr(recent_macd, 'iloc') else recent_macd[i]))
            
            if (recent_closes.iloc[i] if hasattr(recent_closes, 'iloc') else recent_closes[i]) < \
               min(recent_closes.iloc[i-2:i+3] if hasattr(recent_closes, 'iloc') else recent_closes[i-2:i+3]):
                swing_lows.append((i, recent_closes.iloc[i] if hasattr(recent_closes, 'iloc') else recent_closes[i], 
                                 recent_rsi.iloc[i] if hasattr(recent_rsi, 'iloc') else recent_rsi[i],
                                 recent_macd.iloc[i] if hasattr(recent_macd, 'iloc') else recent_macd[i]))
        
        # Kiểm tra Hidden Divergence (ngược với Regular Divergence)
        if len(swing_highs) >= 2:
            # Hidden Bearish Divergence (xu hướng giảm tiếp diễn)
            last_high = swing_highs[-1]
            prev_high = swing_highs[-2]
            
            if last_high[1] < prev_high[1] and last_high[2] > prev_high[2]:
                return {
                    'type': 'hidden_bearish_divergence',
                    'indicator': 'RSI_MACD',
                    'strength': 0.7,
                    'signal': 'Short',
                    'analysis': f'Hidden Bearish Divergence: Giá thấp hơn nhưng RSI cao hơn - xu hướng giảm tiếp diễn'
                }
        
        if len(swing_lows) >= 2:
            # Hidden Bullish Divergence (xu hướng tăng tiếp diễn)
            last_low = swing_lows[-1]
            prev_low = swing_lows[-2]
            
            if last_low[1] > prev_low[1] and last_low[2] < prev_low[2]:
                return {
                    'type': 'hidden_bullish_divergence',
                    'indicator': 'RSI_MACD',
                    'strength': 0.7,
                    'signal': 'Long',
                    'analysis': f'Hidden Bullish Divergence: Giá cao hơn nhưng RSI thấp hơn - xu hướng tăng tiếp diễn'
                }
        
        return None
    except Exception as e:
        logger.error(f"Lỗi phân tích Hidden Divergence: {e}")
        return None

def analyze_multi_timeframe_ema_system(symbol, current_price):
    """Phân tích hệ thống EMA đa khung thời gian theo phương pháp trading Việt Nam"""
    try:
        multi_tf_analysis = {
            'h4_trend': 'neutral',
            'h1_trend': 'neutral',
            'm15_trend': 'neutral',
            'entry_signal': 'hold',
            'trend_alignment': False,
            'analysis': ''
        }
        
        # Lấy dữ liệu cho các khung thời gian
        timeframes = ['4h', '1h', '15m']
        tf_data = {}
        
        for tf in timeframes:
            try:
                data = load_or_fetch_historical_data(symbol, tf)
                if data and len(data['close']) >= 50:
                    close = pd.Series(data['close'])
                    ema34 = ta.trend.ema_indicator(close, window=34)
                    ema89 = ta.trend.ema_indicator(close, window=89)
                    
                    tf_data[tf] = {
                        'close': close,
                        'ema34': ema34,
                        'ema89': ema89,
                        'current_price': current_price
                    }
            except Exception as e:
                logger.error(f"Lỗi lấy dữ liệu {tf} cho {symbol}: {e}")
                continue
        
        # Phân tích xu hướng H4 (khung chính)
        if '4h' in tf_data:
            h4_data = tf_data['4h']
            h4_close = h4_data['close'].iloc[-1]
            h4_ema34 = h4_data['ema34'].iloc[-1]
            h4_ema89 = h4_data['ema89'].iloc[-1]
            
            if h4_close > h4_ema34 and h4_ema34 > h4_ema89:
                multi_tf_analysis['h4_trend'] = 'bullish'
            elif h4_close < h4_ema34 and h4_ema34 < h4_ema89:
                multi_tf_analysis['h4_trend'] = 'bearish'
            else:
                multi_tf_analysis['h4_trend'] = 'mixed'
        
        # Phân tích xu hướng H1 (khung vào lệnh)
        if '1h' in tf_data:
            h1_data = tf_data['1h']
            h1_close = h1_data['close'].iloc[-1]
            h1_ema34 = h1_data['ema34'].iloc[-1]
            h1_ema89 = h1_data['ema89'].iloc[-1]
            
            if h1_close > h1_ema34 and h1_ema34 > h1_ema89:
                multi_tf_analysis['h1_trend'] = 'bullish'
            elif h1_close < h1_ema34 and h1_ema34 < h1_ema89:
                multi_tf_analysis['h1_trend'] = 'bearish'
            else:
                multi_tf_analysis['h1_trend'] = 'mixed'
        
        # Phân tích xu hướng M15 (khung vào lệnh chi tiết)
        if '15m' in tf_data:
            m15_data = tf_data['15m']
            m15_close = m15_data['close'].iloc[-1]
            m15_ema34 = m15_data['ema34'].iloc[-1]
            m15_ema89 = m15_data['ema89'].iloc[-1]
            
            if m15_close > m15_ema34 and m15_ema34 > m15_ema89:
                multi_tf_analysis['m15_trend'] = 'bullish'
            elif m15_close < m15_ema34 and m15_ema34 < m15_ema89:
                multi_tf_analysis['m15_trend'] = 'bearish'
            else:
                multi_tf_analysis['m15_trend'] = 'mixed'
        
        # Xác định tín hiệu entry theo quy tắc đa khung thời gian
        if multi_tf_analysis['h4_trend'] == 'bullish':
            if multi_tf_analysis['h1_trend'] == 'bullish' and multi_tf_analysis['m15_trend'] == 'bullish':
                multi_tf_analysis['entry_signal'] = 'buy'
                multi_tf_analysis['trend_alignment'] = True
                multi_tf_analysis['analysis'] = 'Tất cả khung thời gian đồng thuận tăng - tín hiệu mua mạnh'
            elif multi_tf_analysis['h1_trend'] == 'bullish' and multi_tf_analysis['m15_trend'] == 'mixed':
                multi_tf_analysis['entry_signal'] = 'buy'
                multi_tf_analysis['trend_alignment'] = True
                multi_tf_analysis['analysis'] = 'H4 và H1 tăng, M15 hỗn hợp - tín hiệu mua'
            else:
                multi_tf_analysis['entry_signal'] = 'hold'
                multi_tf_analysis['analysis'] = 'H4 tăng nhưng H1/M15 không đồng thuận - chờ'
        
        elif multi_tf_analysis['h4_trend'] == 'bearish':
            if multi_tf_analysis['h1_trend'] == 'bearish' and multi_tf_analysis['m15_trend'] == 'bearish':
                multi_tf_analysis['entry_signal'] = 'sell'
                multi_tf_analysis['trend_alignment'] = True
                multi_tf_analysis['analysis'] = 'Tất cả khung thời gian đồng thuận giảm - tín hiệu bán mạnh'
            elif multi_tf_analysis['h1_trend'] == 'bearish' and multi_tf_analysis['m15_trend'] == 'mixed':
                multi_tf_analysis['entry_signal'] = 'sell'
                multi_tf_analysis['trend_alignment'] = True
                multi_tf_analysis['analysis'] = 'H4 và H1 giảm, M15 hỗn hợp - tín hiệu bán'
            else:
                multi_tf_analysis['entry_signal'] = 'hold'
                multi_tf_analysis['analysis'] = 'H4 giảm nhưng H1/M15 không đồng thuận - chờ'
        
        else:
            multi_tf_analysis['entry_signal'] = 'hold'
            multi_tf_analysis['analysis'] = 'H4 hỗn hợp - đứng ngoài quan sát'
        
        return multi_tf_analysis
    except Exception as e:
        logger.error(f"Lỗi phân tích đa khung thời gian EMA: {e}")
        return {'h4_trend': 'neutral', 'h1_trend': 'neutral', 'm15_trend': 'neutral', 'entry_signal': 'hold', 'trend_alignment': False, 'analysis': ''}

def detect_smart_money_accumulation_distribution(close, volume, ema34, ema89):
    """Phát hiện giai đoạn tích lũy và phân phối của smart money"""
    try:
        smart_money_analysis = {
            'phase': 'neutral',
            'accumulation_strength': 0.0,
            'distribution_strength': 0.0,
            'smart_money_signal': 'hold',
            'analysis': ''
        }
        
        if len(close) < 20:
            return smart_money_analysis
        
        # Lấy 20 giá trị gần nhất
        recent_closes = close.iloc[-20:] if hasattr(close, 'iloc') else close[-20:]
        recent_volume = volume.iloc[-20:] if hasattr(volume, 'iloc') else volume[-20:]
        recent_ema34 = ema34.iloc[-20:] if hasattr(ema34, 'iloc') else ema34[-20:]
        recent_ema89 = ema89.iloc[-20:] if hasattr(ema89, 'iloc') else ema89[-20:]
        
        # Tính volume trung bình
        avg_volume = np.mean(recent_volume)
        
        # Phân tích tích lũy (Accumulation)
        accumulation_signals = 0
        for i in range(len(recent_closes)):
            current_close = recent_closes.iloc[i] if hasattr(recent_closes, 'iloc') else recent_closes[i]
            current_volume = recent_volume.iloc[i] if hasattr(recent_volume, 'iloc') else recent_volume[i]
            current_ema34 = recent_ema34.iloc[i] if hasattr(recent_ema34, 'iloc') else recent_ema34[i]
            
            # Tích lũy: giá giảm nhưng volume tăng, giá gần EMA34
            if current_close < current_ema34 and current_volume > avg_volume * 1.2:
                accumulation_signals += 1
        
        # Phân tích phân phối (Distribution)
        distribution_signals = 0
        for i in range(len(recent_closes)):
            current_close = recent_closes.iloc[i] if hasattr(recent_closes, 'iloc') else recent_closes[i]
            current_volume = recent_volume.iloc[i] if hasattr(recent_volume, 'iloc') else recent_volume[i]
            current_ema34 = recent_ema34.iloc[i] if hasattr(recent_ema34, 'iloc') else recent_ema34[i]
            
            # Phân phối: giá tăng nhưng volume giảm, giá gần EMA34
            if current_close > current_ema34 and current_volume < avg_volume * 0.8:
                distribution_signals += 1
        
        # Xác định giai đoạn
        if accumulation_signals >= 3:
            smart_money_analysis['phase'] = 'accumulation'
            smart_money_analysis['accumulation_strength'] = min(accumulation_signals / 10.0, 1.0)
            smart_money_analysis['smart_money_signal'] = 'buy'
            smart_money_analysis['analysis'] = f'Giai đoạn tích lũy - {accumulation_signals} tín hiệu'
        elif distribution_signals >= 3:
            smart_money_analysis['phase'] = 'distribution'
            smart_money_analysis['distribution_strength'] = min(distribution_signals / 10.0, 1.0)
            smart_money_analysis['smart_money_signal'] = 'sell'
            smart_money_analysis['analysis'] = f'Giai đoạn phân phối - {distribution_signals} tín hiệu'
        else:
            smart_money_analysis['phase'] = 'neutral'
            smart_money_analysis['analysis'] = 'Không có dấu hiệu tích lũy/phân phối rõ ràng'
        
        return smart_money_analysis
    except Exception as e:
        logger.error(f"Lỗi phát hiện smart money: {e}")
        return {'phase': 'neutral', 'accumulation_strength': 0.0, 'distribution_strength': 0.0, 'smart_money_signal': 'hold', 'analysis': ''}

def detect_whale_activity(close, volume, ema34, ema89):
    """Phát hiện hoạt động của cá mập (whale)"""
    try:
        whale_analysis = {
            'whale_activity': False,
            'activity_type': 'none',
            'whale_signal': 'hold',
            'analysis': ''
        }
        
        if len(close) < 10:
            return whale_analysis
        
        # Lấy 10 giá trị gần nhất
        recent_closes = close.iloc[-10:] if hasattr(close, 'iloc') else close[-10:]
        recent_volume = volume.iloc[-10:] if hasattr(volume, 'iloc') else volume[-10:]
        recent_ema34 = ema34.iloc[-10:] if hasattr(ema34, 'iloc') else ema34[-10:]
        
        # Tính volume trung bình
        avg_volume = np.mean(recent_volume)
        
        # Phát hiện volume spike (tăng đột biến)
        volume_spikes = 0
        for i in range(len(recent_volume)):
            current_volume = recent_volume.iloc[i] if hasattr(recent_volume, 'iloc') else recent_volume[i]
            if current_volume > avg_volume * 3:  # Volume tăng gấp 3 lần
                volume_spikes += 1
        
        # Phát hiện giá bị đẩy mạnh
        price_moves = 0
        for i in range(1, len(recent_closes)):
            prev_close = recent_closes.iloc[i-1] if hasattr(recent_closes, 'iloc') else recent_closes[i-1]
            current_close = recent_closes.iloc[i] if hasattr(recent_closes, 'iloc') else recent_closes[i]
            price_change = abs(current_close - prev_close) / prev_close
            
            if price_change > 0.05:  # Giá thay đổi > 5%
                price_moves += 1
        
        # Xác định hoạt động cá mập
        if volume_spikes >= 2 and price_moves >= 2:
            whale_analysis['whale_activity'] = True
            whale_analysis['activity_type'] = 'strong'
            whale_analysis['whale_signal'] = 'watch'
            whale_analysis['analysis'] = f'Hoạt động cá mập mạnh - {volume_spikes} volume spikes, {price_moves} price moves'
        elif volume_spikes >= 1 or price_moves >= 1:
            whale_analysis['whale_activity'] = True
            whale_analysis['activity_type'] = 'moderate'
            whale_analysis['whale_signal'] = 'watch'
            whale_analysis['analysis'] = f'Hoạt động cá mập vừa - {volume_spikes} volume spikes, {price_moves} price moves'
        
        return whale_analysis
    except Exception as e:
        logger.error(f"Lỗi phát hiện hoạt động cá mập: {e}")
        return {'whale_activity': False, 'activity_type': 'none', 'whale_signal': 'hold', 'analysis': ''}

def analyze_crypto_correlation_ml(btc_data, eth_data, btc_d_data):
    """Phân tích mối liên kết giữa BTC, ETH và BTC.D bằng AI/ML"""
    try:
        correlation_analysis = {
            'btc_eth_correlation': 0.0,
            'btc_dominance_impact': 0.0,
            'market_sentiment': 'neutral',
            'btc_prediction_confidence': 0.0,
            'analysis': '',
            'ml_signals': []
        }
        
        if not all([btc_data, eth_data, btc_d_data]):
            return correlation_analysis
        
        # Chuyển đổi dữ liệu sang pandas DataFrame
        btc_df = pd.DataFrame(btc_data)
        eth_df = pd.DataFrame(eth_data)
        btc_d_df = pd.DataFrame(btc_d_data)
        
        # Tính correlation giữa BTC và ETH
        if len(btc_df) >= 20 and len(eth_df) >= 20:
            min_len = min(len(btc_df), len(eth_df))
            btc_prices = btc_df['close'].iloc[-min_len:]
            eth_prices = eth_df['close'].iloc[-min_len:]
            
            # Tính correlation
            btc_eth_corr = btc_prices.corr(eth_prices)
            correlation_analysis['btc_eth_correlation'] = btc_eth_corr
            
            # Phân tích BTC Dominance impact
            if len(btc_d_df) >= 20:
                btc_d_prices = btc_d_df['close'].iloc[-min_len:]
                btc_d_corr = btc_prices.corr(btc_d_prices)
                correlation_analysis['btc_dominance_impact'] = btc_d_corr
        
        # Tạo features cho ML
        ml_features = create_correlation_ml_features(btc_df, eth_df, btc_d_df)
        
        # Phân tích bằng ML models
        ml_analysis = analyze_correlation_with_ml(ml_features)
        correlation_analysis.update(ml_analysis)
        
        return correlation_analysis
    except Exception as e:
        logger.error(f"Lỗi phân tích correlation ML: {e}")
        return {'btc_eth_correlation': 0.0, 'btc_dominance_impact': 0.0, 'market_sentiment': 'neutral', 'btc_prediction_confidence': 0.0, 'analysis': '', 'ml_signals': []}

def create_correlation_ml_features(btc_df, eth_df, btc_d_df):
    """Tạo features cho ML từ dữ liệu correlation"""
    try:
        features = {}
        
        # Tính toán các chỉ số correlation
        if len(btc_df) >= 20 and len(eth_df) >= 20:
            min_len = min(len(btc_df), len(eth_df))
            
            # Price correlation
            btc_prices = btc_df['close'].iloc[-min_len:]
            eth_prices = eth_df['close'].iloc[-min_len:]
            features['price_correlation'] = btc_prices.corr(eth_prices)
            
            # Volume correlation
            btc_volume = btc_df['volume'].iloc[-min_len:]
            eth_volume = eth_df['volume'].iloc[-min_len:]
            features['volume_correlation'] = btc_volume.corr(eth_volume)
            
            # Price change correlation
            btc_change = btc_prices.pct_change().dropna()
            eth_change = eth_prices.pct_change().dropna()
            features['change_correlation'] = btc_change.corr(eth_change)
            
            # Volatility correlation
            btc_volatility = btc_change.rolling(7).std()
            eth_volatility = eth_change.rolling(7).std()
            features['volatility_correlation'] = btc_volatility.corr(eth_volatility)
        
        # BTC Dominance analysis
        if len(btc_d_df) >= 20:
            btc_d_prices = btc_d_df['close'].iloc[-min_len:]
            features['btc_dominance_trend'] = btc_d_prices.pct_change().mean()
            features['btc_dominance_volatility'] = btc_d_prices.pct_change().std()
            features['btc_dominance_correlation'] = btc_prices.corr(btc_d_prices)
        
        # Market structure features
        features['btc_eth_ratio'] = btc_prices.iloc[-1] / eth_prices.iloc[-1]
        features['btc_eth_ratio_change'] = (btc_prices.iloc[-1] / eth_prices.iloc[-1]) / (btc_prices.iloc[-5] / eth_prices.iloc[-5]) - 1
        
        return features
    except Exception as e:
        logger.error(f"Lỗi tạo correlation features: {e}")
        return {}

def analyze_correlation_with_ml(features):
    """Phân tích correlation bằng ML models"""
    try:
        ml_analysis = {
            'market_sentiment': 'neutral',
            'btc_prediction_confidence': 0.0,
            'analysis': '',
            'ml_signals': []
        }
        
        if not features:
            return ml_analysis
        
        # Tạo feature vector
        feature_vector = np.array([
            features.get('price_correlation', 0),
            features.get('volume_correlation', 0),
            features.get('change_correlation', 0),
            features.get('volatility_correlation', 0),
            features.get('btc_dominance_trend', 0),
            features.get('btc_dominance_volatility', 0),
            features.get('btc_dominance_correlation', 0),
            features.get('btc_eth_ratio', 0),
            features.get('btc_eth_ratio_change', 0)
        ]).reshape(1, -1)
        
        # Phân tích correlation patterns
        price_corr = features.get('price_correlation', 0)
        dominance_corr = features.get('btc_dominance_correlation', 0)
        dominance_trend = features.get('btc_dominance_trend', 0)
        
        # Xác định market sentiment
        if price_corr > 0.7 and dominance_corr > 0.3:
            ml_analysis['market_sentiment'] = 'strong_correlation'
            ml_analysis['btc_prediction_confidence'] = 0.8
            ml_analysis['analysis'] = 'BTC và ETH có correlation mạnh, BTC Dominance tăng'
            ml_analysis['ml_signals'].append('Strong correlation signal')
        elif price_corr > 0.5 and dominance_corr < -0.2:
            ml_analysis['market_sentiment'] = 'altcoin_season'
            ml_analysis['btc_prediction_confidence'] = 0.6
            ml_analysis['analysis'] = 'Altcoin season - ETH có thể outperform BTC'
            ml_analysis['ml_signals'].append('Altcoin season signal')
        elif price_corr < 0.3:
            ml_analysis['market_sentiment'] = 'divergence'
            ml_analysis['btc_prediction_confidence'] = 0.4
            ml_analysis['analysis'] = 'BTC và ETH có correlation thấp - thị trường phân hóa'
            ml_analysis['ml_signals'].append('Divergence signal')
        else:
            ml_analysis['market_sentiment'] = 'neutral'
            ml_analysis['btc_prediction_confidence'] = 0.5
            ml_analysis['analysis'] = 'Correlation bình thường'
        
        # Phân tích BTC Dominance
        if dominance_trend > 0.02:
            ml_analysis['ml_signals'].append('BTC Dominance increasing')
            ml_analysis['analysis'] += ' - BTC Dominance tăng mạnh'
        elif dominance_trend < -0.02:
            ml_analysis['ml_signals'].append('BTC Dominance decreasing')
            ml_analysis['analysis'] += ' - BTC Dominance giảm'
        
        return ml_analysis
    except Exception as e:
        logger.error(f"Lỗi phân tích ML correlation: {e}")
        return {'market_sentiment': 'neutral', 'btc_prediction_confidence': 0.0, 'analysis': '', 'ml_signals': []}

def get_btc_dominance_data():
    """Lấy dữ liệu BTC Dominance"""
    try:
        # Sử dụng yfinance để lấy dữ liệu BTC Dominance
        import yfinance as yf
        
        # BTC Dominance ticker (sử dụng ticker giả lập vì yfinance không có BTC Dominance trực tiếp)
        # Thay vào đó, chúng ta sẽ tính toán từ dữ liệu BTC và ETH
        btc_ticker = yf.Ticker("BTC-USD")
        eth_ticker = yf.Ticker("ETH-USD")
        
        btc_hist = btc_ticker.history(period="30d", interval="1h")
        eth_hist = eth_ticker.history(period="30d", interval="1h")
        
        if btc_hist.empty or eth_hist.empty:
            return None
        
        # Tính toán BTC Dominance giả lập (BTC market cap / Total crypto market cap)
        # Đây là một approximation đơn giản
        btc_dominance = btc_hist['Close'] / (btc_hist['Close'] + eth_hist['Close']) * 100
        
        # Chuyển đổi sang format giống với dữ liệu crypto khác
        btc_d_data = {
            'open': btc_dominance.values,
            'high': btc_dominance.values,
            'low': btc_dominance.values,
            'close': btc_dominance.values,
            'volume': btc_hist['Volume'].values,
            'timestamp': btc_hist.index.astype(np.int64) // 10**9
        }
        
        return btc_d_data
    except Exception as e:
        logger.error(f"Lỗi lấy dữ liệu BTC Dominance: {e}")
        return None

def enhance_btc_analysis_with_correlation(btc_analysis, correlation_analysis):
    """Tăng cường phân tích BTC với thông tin correlation"""
    try:
        enhanced_analysis = btc_analysis.copy()
        
        # Thêm thông tin correlation
        enhanced_analysis['correlation_analysis'] = correlation_analysis
        
        # Điều chỉnh confidence dựa trên correlation
        original_confidence = enhanced_analysis.get('confidence', 0.5)
        correlation_confidence = correlation_analysis.get('btc_prediction_confidence', 0.5)
        
        # Kết hợp confidence
        combined_confidence = (original_confidence * 0.7) + (correlation_confidence * 0.3)
        enhanced_analysis['confidence'] = combined_confidence
        
        # Điều chỉnh signal dựa trên market sentiment
        market_sentiment = correlation_analysis.get('market_sentiment', 'neutral')
        original_signal = enhanced_analysis.get('signal', 'Hold')
        
        if market_sentiment == 'strong_correlation' and original_signal != 'Hold':
            enhanced_analysis['signal_strength'] = enhanced_analysis.get('signal_strength', 0.5) + 0.2
        elif market_sentiment == 'divergence':
            enhanced_analysis['signal_strength'] = enhanced_analysis.get('signal_strength', 0.5) - 0.1
        
        # Thêm analysis text
        correlation_text = correlation_analysis.get('analysis', '')
        if correlation_text:
            enhanced_analysis['analysis'] += f"\n🔗 Correlation: {correlation_text}"
        
        return enhanced_analysis
    except Exception as e:
        logger.error(f"Lỗi enhance BTC analysis: {e}")
        return btc_analysis

def calculate_divergence_consensus(divergences):
    """Tính toán consensus từ các divergence"""
    if not divergences:
        return {'signal': 'Hold', 'strength': 0, 'count': 0}
    
    long_signals = [d for d in divergences if d['signal'] == 'Long']
    short_signals = [d for d in divergences if d['signal'] == 'Short']
    
    if len(long_signals) > len(short_signals):
        avg_strength = sum(d['strength'] for d in long_signals) / len(long_signals)
        return {
            'signal': 'Long',
            'strength': avg_strength,
            'count': len(long_signals),
            'divergences': long_signals
        }
    elif len(short_signals) > len(long_signals):
        avg_strength = sum(d['strength'] for d in short_signals) / len(short_signals)
        return {
            'signal': 'Short',
            'strength': avg_strength,
            'count': len(short_signals),
            'divergences': short_signals
        }
    else:
        # Nếu số lượng bằng nhau, chọn theo strength cao hơn
        max_long_strength = max([d['strength'] for d in long_signals]) if long_signals else 0
        max_short_strength = max([d['strength'] for d in short_signals]) if short_signals else 0
        
        if max_long_strength > max_short_strength:
            return {
                'signal': 'Long',
                'strength': max_long_strength,
                'count': len(long_signals),
                'divergences': long_signals
            }
        elif max_short_strength > max_long_strength:
            return {
                'signal': 'Short',
                'strength': max_short_strength,
                'count': len(short_signals),
                'divergences': short_signals
            }
        else:
            return {'signal': 'Hold', 'strength': 0, 'count': 0}

def fetch_historical_data_for_ml(symbol, timeframe, limit=None):
    """Lấy dữ liệu lịch sử cho ML training từ Binance API với fallback"""
    try:
        if limit is None:
            limit = ML_HISTORICAL_CANDLES
            
        logger.info(f"📊 Đang lấy dữ liệu lịch sử cho {symbol} ({timeframe}) - {limit} candles...")
        
        if exchange:
            # Lấy dữ liệu từ Binance
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) < ML_MIN_SAMPLES:
                return None
            
            # Chuyển đổi thành DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
        else:
            # Fallback sử dụng yfinance
            symbol_mapping = {
                'BTC/USDT': 'BTC-USD',
                'ETH/USDT': 'ETH-USD'
            }
            
            yf_symbol = symbol_mapping.get(symbol, symbol.replace('/', '-'))
            ticker = yf.Ticker(yf_symbol)
            
            # Chuyển đổi timeframe
            period_mapping = {
                '1h': '1h',
                '2h': '2h', 
                '4h': '4h',
                '6h': '6h',
                '8h': '8h',
                '12h': '12h',
                '1d': '1d',
                '3d': '3d',
                '1w': '1wk'
            }
            
            period = period_mapping.get(timeframe, '1d')
            df = ticker.history(period=f"{limit}d", interval=period)
            
            if len(df) < ML_MIN_SAMPLES:
                return None
        
        # Lưu dữ liệu gốc (thay thế ký tự / bằng _)
        safe_symbol = symbol.replace('/', '_')
        data_file = os.path.join(ML_DATA_DIR, f"{safe_symbol}_{timeframe}_historical.csv")
        df.to_csv(data_file)
        

        
        return {
            'open': df['Open'].values if 'Open' in df.columns else df['open'].values,
            'high': df['High'].values if 'High' in df.columns else df['high'].values,
            'low': df['Low'].values if 'Low' in df.columns else df['low'].values,
            'close': df['Close'].values if 'Close' in df.columns else df['close'].values,
            'volume': df['Volume'].values if 'Volume' in df.columns else df['volume'].values,
            'timestamp': df.index.values
        }
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy dữ liệu lịch sử cho {symbol} ({timeframe}): {e}")
        
        # Thử fallback với yfinance
        try:
            symbol_mapping = {
                'BTC/USDT': 'BTC-USD',
                'ETH/USDT': 'ETH-USD'
            }
            
            yf_symbol = symbol_mapping.get(symbol, symbol.replace('/', '-'))
            ticker = yf.Ticker(yf_symbol)
            period = period_mapping.get(timeframe, '1d')
            df = ticker.history(period=f"{limit}d", interval=period)
            
            if len(df) > 0:

                
                # Lưu dữ liệu gốc
                safe_symbol = symbol.replace('/', '_')
                data_file = os.path.join(ML_DATA_DIR, f"{safe_symbol}_{timeframe}_historical.csv")
                df.to_csv(data_file)
                
                return {
                    'open': df['Open'].values,
                    'high': df['High'].values,
                    'low': df['Low'].values,
                    'close': df['Close'].values,
                    'volume': df['Volume'].values,
                    'timestamp': df.index.values
                }
        except Exception as e2:
            logger.error(f"❌ Fallback yfinance cũng thất bại cho {symbol}: {e2}")
        
        return None

def load_or_fetch_historical_data(symbol, timeframe):
    """Load dữ liệu lịch sử từ file hoặc fetch từ API nếu chưa có"""
    try:
        safe_symbol = symbol.replace('/', '_')
        data_file = os.path.join(ML_DATA_DIR, f"{safe_symbol}_{timeframe}_historical.csv")
        
        # Kiểm tra xem file có tồn tại và còn mới không (trong vòng 24h)
        if os.path.exists(data_file):
            file_time = os.path.getmtime(data_file)
            current_time = time.time()
            
            # Nếu file còn mới (trong vòng 24h), load từ file
            if current_time - file_time < 86400:  # 24 giờ
                df = pd.read_csv(data_file, index_col='timestamp', parse_dates=True)
                
                return {
                    'open': df['open'].values,
                    'high': df['high'].values,
                    'low': df['low'].values,
                    'close': df['close'].values,
                    'volume': df['volume'].values,
                    'timestamp': df.index.values
                }
        
        # Nếu không có file hoặc file cũ, fetch từ API
        return fetch_historical_data_for_ml(symbol, timeframe)
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi load/fetch dữ liệu lịch sử cho {symbol} ({timeframe}): {e}")
        return None

def display_ml_features_info():
    """Hiển thị thông tin về các features ML"""
    features_info = {
        'Technical Indicators (12)': [
            'RSI - Relative Strength Index',
            'MACD - Moving Average Convergence Divergence',
            'MACD Signal - MACD Signal Line',
            'BB Position - Bollinger Bands Position',
            'BB Width - Bollinger Bands Width',
            'EMA Ratio 20/50 - Exponential Moving Average Ratio',
            'MA Ratio 20/50 - Simple Moving Average Ratio',
            'Stoch K - Stochastic Oscillator %K',
            'Stoch D - Stochastic Oscillator %D',
            'ADX - Average Directional Index',
            'ATR - Average True Range',
            'OBV - On Balance Volume'
        ],
        'Price Features (8)': [
            'Price Change - Percentage Change',
            'High/Low Ratio - High to Low Price Ratio',
            'Close/Open Ratio - Close to Open Price Ratio',
            'RSI Oversold - RSI Below 30',
            'RSI Overbought - RSI Above 70',
            'MACD Cross - MACD Above Signal',
            'MACD Histogram - MACD - Signal',
            'Momentum - 5-period Price Change',
            'Rate of Change - 5-period Percentage Change',
            'Volatility Ratio - Standard Deviation Ratio'
        ],
        'Volume Features (3)': [
            'Volume Ratio - Current Volume to MA',
            'Volume Z-Score - Volume Standardization',
            'Volume Price Trend - Volume * Price Change'
        ],
        'Support/Resistance (6)': [
            'Support Distance - Distance to Support Level',
            'Resistance Distance - Distance to Resistance Level',
            'Fibonacci Retracement Levels - 23.6%, 38.2%, 50%, 61.8%',
            'Pivot Points - Classic Pivot, R1, S1, R2, S2, R3, S3',
            'Swing Highs/Lows - Dynamic Support/Resistance',
            'Volume Weighted Levels - High Volume Price Zones'
        ],
        'Market Structure (2)': [
            'Trend Strength - Higher Highs vs Lower Lows',
            'Price Acceleration - Rate of Price Change'
        ],
        'Price Patterns (2)': [
            'Hammer - Hammer Candlestick Pattern',
            'Doji - Doji Candlestick Pattern'
        ],
        'Advanced Support/Resistance (4)': [
            'Psychological Levels - Round Numbers (1000, 2000, 5000)',
            'Historical Levels - Cluster Analysis of Peaks/Troughs',
            'Support/Resistance Strength Analysis - Risk Assessment',
            'Breakout Potential - Direction and Probability Analysis'
        ]
    }
    
    print("\n🤖 MACHINE LEARNING FEATURES (40+ Features)")
    print("=" * 50)
    
    total_features = 0
    for category, features in features_info.items():
        print(f"\n📊 {category}:")
        for feature in features:
            print(f"   • {feature}")
        total_features += len(features)
    
    print(f"\n📈 Tổng cộng: {total_features} features")
    print("🎯 Target: Next period price direction (1 = Up, 0 = Down)")
    print("📊 Training Data: 5000 historical candles from Binance API")
    print("🤖 Models: Random Forest, XGBoost, LightGBM, Gradient Boosting, Logistic Regression, SVM")
    print("🔄 Auto-training: Every 24 hours")
    print("💾 Data Storage: Historical data cached locally")
    print("🎯 Advanced Support/Resistance: Fibonacci, Pivot Points, Swing Levels, Volume Analysis, Psychological Levels")

def get_ml_training_status():
    """Kiểm tra trạng thái training ML models và dữ liệu"""
    try:
        ensure_ml_directories()
        
        status = {
            'models_trained': [],
            'models_missing': [],
            'last_training': None,
            'data_files': [],
            'data_statistics': {}
        }
        
        # Kiểm tra models đã train
        for symbol in ['BTC/USDT', 'ETH/USDT']:
            for timeframe in ML_TIMEFRAMES:
                model_files = []
                for model_type in ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting', 'logistic_regression', 'svm']:
                    safe_symbol = symbol.replace('/', '_')
                    model_file = os.path.join(ML_MODELS_DIR, f"{safe_symbol}_{timeframe}_{model_type}.joblib")
                    if os.path.exists(model_file):
                        model_files.append(model_type)
                
                if model_files:
                    status['models_trained'].append(f"{symbol} ({timeframe}): {len(model_files)} models")
                else:
                    status['models_missing'].append(f"{symbol} ({timeframe})")
                
                # Thêm thống kê dữ liệu
                stats = get_data_statistics(symbol, timeframe)
                if stats:
                    status['data_statistics'][f"{symbol}_{timeframe}"] = stats
        
        # Kiểm tra data files
        data_files = [f for f in os.listdir(ML_DATA_DIR) if f.endswith('_historical.csv')]
        status['data_files'] = data_files
        
        return status
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi kiểm tra trạng thái ML: {e}")
        return None

def display_data_update_summary():
    """Hiển thị tóm tắt về việc cập nhật dữ liệu"""
    try:
        logger.info("📊 TÓM TẮT CẬP NHẬT DỮ LIỆU:")
        logger.info("=" * 50)
        
        total_files = 0
        total_candles = 0
        total_size_mb = 0
        
        for symbol in ['BTC/USDT', 'ETH/USDT']:
            for timeframe in ML_TIMEFRAMES:
                stats = get_data_statistics(symbol, timeframe)
                if stats:
                    total_files += 1
                    total_candles += stats['total_candles']
                    total_size_mb += stats['file_size_mb']
                    
                    is_fresh, freshness_msg = check_data_freshness(symbol, timeframe)
                    status_emoji = "✅" if is_fresh else "⚠️"
                    
                    logger.info(f"{status_emoji} {symbol} ({timeframe}): {stats['total_candles']} candles, {stats['file_size_mb']:.2f}MB")
                    logger.info(f"   📅 {stats['date_range']['start']} → {stats['date_range']['end']}")
                    logger.info(f"   🕒 {freshness_msg}")
                else:
                    logger.info(f"❌ {symbol} ({timeframe}): Chưa có dữ liệu")
        
        logger.info("=" * 50)
        logger.info(f"📈 Tổng cộng: {total_files} files, {total_candles:,} candles, {total_size_mb:.2f}MB")
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi hiển thị tóm tắt dữ liệu: {e}")

def get_latest_timestamp_from_data(data):
    """Lấy timestamp mới nhất từ dữ liệu"""
    try:
        if 'timestamp' in data and len(data['timestamp']) > 0:
            latest_ts = data['timestamp'][-1]
            # Chuyển đổi về pandas Timestamp để tránh lỗi numpy
            if hasattr(latest_ts, 'to_pydatetime'):
                return pd.Timestamp(latest_ts.to_pydatetime())
            elif isinstance(latest_ts, str):
                return pd.to_datetime(latest_ts)
            else:
                return pd.Timestamp(latest_ts)
        return None
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy timestamp mới nhất: {e}")
        return None

def fetch_incremental_data(symbol, timeframe, since_timestamp=None, limit=1000):
    """Lấy dữ liệu tăng dần từ timestamp cụ thể"""
    try:
        logger.info(f"📊 Đang lấy dữ liệu tăng dần cho {symbol} ({timeframe}) từ {since_timestamp}...")
        
        if exchange:
            # Lấy dữ liệu từ Binance với since parameter
            if since_timestamp:
                # Chuyển đổi timestamp thành Unix timestamp (milliseconds) cho Binance API
                if isinstance(since_timestamp, str):
                    since_timestamp = pd.to_datetime(since_timestamp)
                if hasattr(since_timestamp, 'timestamp'):
                    since_ms = int(since_timestamp.timestamp() * 1000)
                else:
                    since_ms = int(since_timestamp * 1000)
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
            else:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return None
            
            # Chuyển đổi thành DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
        else:
            # Fallback sử dụng yfinance
            symbol_mapping = {
                'BTC/USDT': 'BTC-USD',
                'ETH/USDT': 'ETH-USD'
            }
            
            yf_symbol = symbol_mapping.get(symbol, symbol.replace('/', '-'))
            ticker = yf.Ticker(yf_symbol)
            
            # Chuyển đổi timeframe
            period_mapping = {
                '1h': '1h',
                '2h': '2h', 
                '4h': '4h',
                '6h': '6h',
                '8h': '8h',
                '12h': '12h',
                '1d': '1d',
                '3d': '3d',
                '1w': '1wk'
            }
            
            period = period_mapping.get(timeframe, '1d')
            
            # Tính toán period dựa trên since_timestamp
            if since_timestamp:
                # Chuyển đổi timestamp thành datetime
                since_dt = pd.to_datetime(since_timestamp)
                current_dt = pd.Timestamp.now()
                # Chuyển đổi về pandas Timestamp để tránh lỗi numpy
                if hasattr(since_dt, 'to_pydatetime'):
                    since_dt = pd.Timestamp(since_dt.to_pydatetime())
                if hasattr(current_dt, 'to_pydatetime'):
                    current_dt = pd.Timestamp(current_dt.to_pydatetime())
                days_diff = (current_dt - since_dt).days
                period_str = f"{max(days_diff + 1, 1)}d"
            else:
                period_str = f"{limit}d"
            
            df = ticker.history(period=period_str, interval=period)
            
            if len(df) == 0:
                return None
        
        return {
            'open': df['Open'].values if 'Open' in df.columns else df['open'].values,
            'high': df['High'].values if 'High' in df.columns else df['high'].values,
            'low': df['Low'].values if 'Low' in df.columns else df['low'].values,
            'close': df['Close'].values if 'Close' in df.columns else df['close'].values,
            'volume': df['Volume'].values if 'Volume' in df.columns else df['volume'].values,
            'timestamp': df.index.values
        }
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy dữ liệu tăng dần cho {symbol} ({timeframe}): {e}")
        return None

def merge_historical_data(existing_data, new_data):
    """Merge dữ liệu mới với dữ liệu cũ, loại bỏ duplicates"""
    try:
        if existing_data is None or new_data is None:
            return new_data if new_data else existing_data
        
        # Chuyển đổi thành DataFrame để dễ xử lý
        existing_df = pd.DataFrame({
            'open': existing_data['open'],
            'high': existing_data['high'],
            'low': existing_data['low'],
            'close': existing_data['close'],
            'volume': existing_data['volume']
        }, index=existing_data['timestamp'])
        
        new_df = pd.DataFrame({
            'open': new_data['open'],
            'high': new_data['high'],
            'low': new_data['low'],
            'close': new_data['close'],
            'volume': new_data['volume']
        }, index=new_data['timestamp'])
        
        # Merge và loại bỏ duplicates
        merged_df = pd.concat([existing_df, new_df])
        merged_df = merged_df[~merged_df.index.duplicated(keep='last')]  # Giữ dữ liệu mới nhất
        merged_df = merged_df.sort_index()  # Sắp xếp theo thời gian
        
        # Giữ lại tất cả dữ liệu lịch sử để AI/ML học liên tục
        # Không giới hạn số lượng candles - để ML có thể học từ toàn bộ lịch sử
        
        return {
            'open': merged_df['open'].values,
            'high': merged_df['high'].values,
            'low': merged_df['low'].values,
            'close': merged_df['close'].values,
            'volume': merged_df['volume'].values,
            'timestamp': merged_df.index.values
        }
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi merge dữ liệu: {e}")
        return existing_data if existing_data else new_data

def load_and_update_historical_data(symbol, timeframe, force_full_update=False):
    """Load dữ liệu hiện có và cập nhật với dữ liệu mới"""
    try:
        safe_symbol = symbol.replace('/', '_')
        data_file = os.path.join(ML_DATA_DIR, f"{safe_symbol}_{timeframe}_historical.csv")
        
        existing_data = None
        latest_timestamp = None
        
        # Load dữ liệu hiện có nếu có
        if os.path.exists(data_file) and not force_full_update:
            try:
                df = pd.read_csv(data_file, index_col='timestamp', parse_dates=True)
                if len(df) > 0:
                    existing_data = {
                        'open': df['open'].values,
                        'high': df['high'].values,
                        'low': df['low'].values,
                        'close': df['close'].values,
                        'volume': df['volume'].values,
                        'timestamp': df.index.values
                    }
                    latest_timestamp = get_latest_timestamp_from_data(existing_data)
                    logger.info(f"📁 Loaded {len(df)} existing candles for {symbol} ({timeframe})")
            except Exception as e:
                logger.warning(f"⚠️ Lỗi khi load dữ liệu cũ cho {symbol} ({timeframe}): {e}")
        
        # Lấy dữ liệu mới
        if force_full_update:
            logger.info(f"🔄 Force full update for {symbol} ({timeframe})")
            new_data = fetch_historical_data_for_ml(symbol, timeframe)
        else:
            new_data = fetch_incremental_data(symbol, timeframe, latest_timestamp)
        
        if new_data is None:
            logger.warning(f"⚠️ Không thể lấy dữ liệu mới cho {symbol} ({timeframe})")
            return existing_data
        
        # Merge dữ liệu
        if existing_data:
            merged_data = merge_historical_data(existing_data, new_data)
            logger.info(f"🔄 Merged data: {len(existing_data['close'])} existing + {len(new_data['close'])} new = {len(merged_data['close'])} total")
        else:
            merged_data = new_data
            logger.info(f"📊 New data: {len(merged_data['close'])} candles")
        
        # Lưu dữ liệu đã merge
        if merged_data:
            df_to_save = pd.DataFrame({
                'timestamp': merged_data['timestamp'],
                'open': merged_data['open'],
                'high': merged_data['high'],
                'low': merged_data['low'],
                'close': merged_data['close'],
                'volume': merged_data['volume']
            })
            
            df_to_save.to_csv(data_file, index=False)
            logger.info(f"💾 Saved {len(merged_data['close'])} candles to {data_file}")
        
        return merged_data
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi load và cập nhật dữ liệu cho {symbol} ({timeframe}): {e}")
        return None

def check_data_freshness(symbol, timeframe, max_age_hours=24):
    """Kiểm tra độ mới của dữ liệu"""
    try:
        safe_symbol = symbol.replace('/', '_')
        data_file = os.path.join(ML_DATA_DIR, f"{safe_symbol}_{timeframe}_historical.csv")
        
        if not os.path.exists(data_file):
            return False, "File không tồn tại"
        
        file_time = os.path.getmtime(data_file)
        current_time = time.time()
        age_hours = (current_time - file_time) / 3600
        
        if age_hours > max_age_hours:
            return False, f"Dữ liệu cũ ({age_hours:.1f} giờ)"
        
        return True, f"Dữ liệu mới ({age_hours:.1f} giờ)"
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi kiểm tra độ mới dữ liệu: {e}")
        return False, f"Lỗi: {e}"

def get_data_statistics(symbol, timeframe):
    """Lấy thống kê về dữ liệu"""
    try:
        safe_symbol = symbol.replace('/', '_')
        data_file = os.path.join(ML_DATA_DIR, f"{safe_symbol}_{timeframe}_historical.csv")
        
        if not os.path.exists(data_file):
            return None
        
        df = pd.read_csv(data_file, index_col='timestamp', parse_dates=True)
        
        stats = {
            'total_candles': len(df),
            'date_range': {
                'start': df.index.min().strftime('%Y-%m-%d %H:%M'),
                'end': df.index.max().strftime('%Y-%m-%d %H:%M')
            },
            'file_size_mb': os.path.getsize(data_file) / (1024 * 1024),
            'last_updated': datetime.fromtimestamp(os.path.getmtime(data_file)).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy thống kê dữ liệu: {e}")
        return None

def save_ml_prediction(symbol, timeframe, prediction_data, confidence, model_type):
    """Lưu dự đoán của ML để đánh giá độ chính xác sau này"""
    try:
        # Xử lý symbol để tránh lỗi đường dẫn (thay / bằng _)
        safe_symbol = symbol.replace('/', '_')
        predictions_file = os.path.join(ML_DATA_DIR, f"{safe_symbol}_{timeframe}_predictions.csv")
        
        # Lấy giá hiện tại làm entry price
        current_price = prediction_data.get('current_price', 0)
        
        # Tạo dữ liệu dự đoán với thông tin TP/SL
        prediction_record = {
            'timestamp': pd.Timestamp.now(),
            'symbol': symbol,
            'timeframe': timeframe,
            'predicted_price': prediction_data.get('predicted_price', 0),
            'predicted_direction': prediction_data.get('predicted_direction', 'unknown'),
            'confidence': confidence,
            'model_type': model_type,
            'features_used': str(prediction_data.get('features', [])),
            'model_accuracy': prediction_data.get('model_accuracy', 0),
            'prediction_horizon': prediction_data.get('prediction_horizon', '1h'),
            'status': 'pending',  # pending, verified, failed, expired
            'entry_price': current_price,  # Giá vào lệnh
            'target_profit_pct': prediction_data.get('target_profit_pct', 2.0),  # Mục tiêu lợi nhuận 2%
            'stop_loss_pct': prediction_data.get('stop_loss_pct', 1.0),  # Cắt lỗ 1%
            'max_hold_time': prediction_data.get('max_hold_time', '4h')  # Thời gian giữ lệnh tối đa
        }
        
        # Load dữ liệu cũ hoặc tạo mới
        if os.path.exists(predictions_file):
            df = pd.read_csv(predictions_file)
            df = pd.concat([df, pd.DataFrame([prediction_record])], ignore_index=True)
        else:
            df = pd.DataFrame([prediction_record])
        
        # Lưu file
        df.to_csv(predictions_file, index=False)
        
        prediction_id = len(df)
        logger.info(f"💾 Đã lưu dự đoán #{prediction_id} cho {symbol} ({timeframe}): {prediction_data.get('predicted_direction')} - Confidence: {confidence:.2%}")
        
        return prediction_id
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lưu dự đoán ML: {e}")
        return None

def verify_ml_predictions(symbol, timeframe, current_price, current_timestamp):
    """Xác minh dự đoán ML dựa trên xu hướng thực tế thay vì so sánh giá đơn giản"""
    try:
        # Xử lý symbol để tránh lỗi đường dẫn
        safe_symbol = symbol.replace('/', '_')
        predictions_file = os.path.join(ML_DATA_DIR, f"{safe_symbol}_{timeframe}_predictions.csv")
        
        if not os.path.exists(predictions_file):
            return None
        
        df = pd.read_csv(predictions_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Chỉ xem xét các dự đoán pending và đã đến thời gian kiểm tra
        df['prediction_horizon_hours'] = df['prediction_horizon'].map({'1h': 1, '4h': 4, '1d': 24, '1w': 168})
        df['check_time'] = df['timestamp'] + pd.to_timedelta(df['prediction_horizon_hours'], unit='h')
        
        # Lọc dự đoán cần kiểm tra
        pending_predictions = df[
            (df['status'] == 'pending') & 
            (df['check_time'] <= current_timestamp)
        ].copy()
        
        if len(pending_predictions) == 0:
            return None
        
        verified_count = 0
        failed_count = 0
        
        for idx, pred in pending_predictions.iterrows():
            predicted_direction = pred['predicted_direction']
            entry_price = pred['entry_price']
            target_profit_pct = pred.get('target_profit_pct', 2.0)  # Mặc định 2%
            stop_loss_pct = pred.get('stop_loss_pct', 1.0)          # Mặc định 1%
            
            # Tính toán mức TP và SL
            target_profit_price = entry_price * (1 + target_profit_pct / 100)
            stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
            
            # Lấy dữ liệu giá từ thời điểm dự đoán đến hiện tại
            price_data = get_price_data_since_prediction(symbol, timeframe, pred['timestamp'], current_timestamp)
            
            if price_data is None or len(price_data) == 0:
                logger.warning(f"⚠️ Không thể lấy dữ liệu giá cho dự đoán #{pred.name}")
                continue
            
            # Xác định xu hướng thực tế và kết quả giao dịch
            actual_result = determine_actual_trading_result(
                price_data, 
                predicted_direction, 
                entry_price, 
                target_profit_price, 
                stop_loss_price
            )
            
            # Cập nhật trạng thái dựa trên kết quả thực tế
            if actual_result['result'] == 'profit':
                new_status = 'verified'
                accuracy = 1.0
                verified_count += 1
                logger.info(f"✅ Dự đoán #{pred.name} đúng: {predicted_direction} → Chạm TP {target_profit_pct}%")
            elif actual_result['result'] == 'loss':
                new_status = 'failed'
                accuracy = 0.0
                failed_count += 1
                logger.warning(f"❌ Dự đoán #{pred.name} sai: {predicted_direction} → Chạm SL {stop_loss_pct}%")
            else:  # sideways hoặc chưa chạm TP/SL
                new_status = 'expired'
                accuracy = 0.5  # Độ chính xác trung bình
                logger.info(f"⏰ Dự đoán #{pred.name} hết hạn: {predicted_direction} → Không chạm TP/SL")
            
            # Cập nhật thông tin
            df.loc[idx, 'status'] = new_status
            df.loc[idx, 'actual_price'] = current_price
            df.loc[idx, 'verification_time'] = current_timestamp
            df.loc[idx, 'accuracy'] = accuracy
            df.loc[idx, 'actual_result'] = actual_result['result']
            df.loc[idx, 'max_price_reached'] = actual_result['max_price']
            df.loc[idx, 'min_price_reached'] = actual_result['min_price']
            df.loc[idx, 'price_movement_pct'] = actual_result['price_movement_pct']
        
        # Lưu cập nhật
        df.to_csv(predictions_file, index=False)
        
        if verified_count > 0 or failed_count > 0:
            logger.info(f"🔍 Đã xác minh {verified_count + failed_count} dự đoán: {verified_count} đúng, {failed_count} sai")
        
        return {
            'verified': verified_count,
            'failed': failed_count,
            'expired': len(pending_predictions) - verified_count - failed_count,
            'total_checked': len(pending_predictions)
        }
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi xác minh dự đoán ML: {e}")
        return None

def get_price_data_since_prediction(symbol, timeframe, prediction_time, current_time):
    """Lấy dữ liệu giá từ thời điểm dự đoán đến hiện tại"""
    try:
        # Đọc dữ liệu lịch sử đã lưu
        data_file = os.path.join(ML_DATA_DIR, f"{symbol}_{timeframe}_historical.csv")
        if not os.path.exists(data_file):
            return None
        
        df = pd.read_csv(data_file, index_col='timestamp', parse_dates=True)
        
        # Lọc dữ liệu từ thời điểm dự đoán đến hiện tại
        mask = (df.index >= prediction_time) & (df.index <= current_time)
        filtered_data = df[mask].copy()
        
        if len(filtered_data) == 0:
            return None
        
        return filtered_data
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy dữ liệu giá: {e}")
        return None

def determine_actual_trading_result(price_data, predicted_direction, entry_price, target_profit_price, stop_loss_price):
    """Xác định kết quả giao dịch thực tế dựa trên dữ liệu giá"""
    try:
        if len(price_data) == 0:
            return {
                'result': 'unknown',
                'max_price': entry_price,
                'min_price': entry_price,
                'price_movement_pct': 0.0
            }
        
        # Lấy giá cao nhất và thấp nhất trong khoảng thời gian
        max_price = price_data['high'].max()
        min_price = price_data['low'].min()
        
        # Tính phần trăm thay đổi giá
        price_movement_pct = ((max_price - min_price) / entry_price) * 100
        
        # Kiểm tra xem có chạm TP hoặc SL không
        hit_tp = max_price >= target_profit_price
        hit_sl = min_price <= stop_loss_price
        
        # Xác định kết quả
        if hit_tp:
            result = 'profit'
        elif hit_sl:
            result = 'loss'
        else:
            result = 'sideways'  # Không chạm TP/SL
        
        return {
            'result': result,
            'max_price': max_price,
            'min_price': min_price,
            'price_movement_pct': price_movement_pct,
            'hit_tp': hit_tp,
            'hit_sl': hit_sl
        }
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi xác định kết quả giao dịch: {e}")
        return {
            'result': 'unknown',
            'max_price': entry_price,
            'min_price': entry_price,
            'price_movement_pct': 0.0
        }

def get_prediction_accuracy_stats(symbol, timeframe, days_back=30):
    """Lấy thống kê độ chính xác dự đoán ML"""
    try:
        # Xử lý symbol để tránh lỗi đường dẫn
        safe_symbol = symbol.replace('/', '_')
        predictions_file = os.path.join(ML_DATA_DIR, f"{safe_symbol}_{timeframe}_predictions.csv")
        
        if not os.path.exists(predictions_file):
            return None
        
        df = pd.read_csv(predictions_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Lọc dữ liệu theo thời gian
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_back)
        recent_df = df[df['timestamp'] >= cutoff_date]
        
        if len(recent_df) == 0:
            return None
        
        # Tính toán thống kê
        total_predictions = len(recent_df)
        verified_predictions = recent_df[recent_df['status'] == 'verified']
        failed_predictions = recent_df[recent_df['status'] == 'failed']
        expired_predictions = recent_df[recent_df['status'] == 'expired']
        
        # Tính độ chính xác (chỉ tính verified vs failed, không tính expired)
        completed_predictions = len(verified_predictions) + len(failed_predictions)
        accuracy = len(verified_predictions) / completed_predictions if completed_predictions > 0 else 0
        
        # Thống kê theo model type - sử dụng confidence thay vì accuracy
        model_stats = recent_df.groupby('model_type').agg({
            'confidence': ['count', 'mean', 'sum']
        }).round(3)
        
        # Thống kê theo confidence level
        confidence_bins = [0, 0.5, 0.7, 0.9, 1.0]
        confidence_labels = ['Low (0-50%)', 'Medium (50-70%)', 'High (70-90%)', 'Very High (90-100%)']
        recent_df['confidence_bin'] = pd.cut(recent_df['confidence'], bins=confidence_bins, labels=confidence_labels)
        
        confidence_stats = recent_df.groupby('confidence_bin').agg({
            'confidence': ['count', 'mean']
        }).round(3)
        
        return {
            'total_predictions': total_predictions,
            'verified_predictions': len(verified_predictions),
            'failed_predictions': len(failed_predictions),
            'expired_predictions': len(expired_predictions),
            'completed_predictions': completed_predictions,
            'overall_accuracy': accuracy,
            'model_type_stats': model_stats.to_dict(),
            'confidence_stats': confidence_stats.to_dict(),
            'period_days': days_back
        }
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy thống kê độ chính xác: {e}")
        return None

def adjust_ml_algorithm_based_on_accuracy(symbol, timeframe, current_prediction):
    """Điều chỉnh thuật toán ML dựa trên độ chính xác lịch sử"""
    try:
        accuracy_stats = get_prediction_accuracy_stats(symbol, timeframe, days_back=7)
        
        if not accuracy_stats:
            return current_prediction
        
        overall_accuracy = accuracy_stats['overall_accuracy']
        confidence = current_prediction.get('confidence', 0.5)
        
        # Điều chỉnh confidence dựa trên độ chính xác
        if overall_accuracy > 0.7:  # Độ chính xác cao
            adjusted_confidence = min(confidence * 1.1, 1.0)
            adjustment_reason = f"Tăng confidence do độ chính xác cao ({overall_accuracy:.1%})"
        elif overall_accuracy < 0.4:  # Độ chính xác thấp
            adjusted_confidence = max(confidence * 0.8, 0.1)
            adjustment_reason = f"Giảm confidence do độ chính xác thấp ({overall_accuracy:.1%})"
        else:
            adjusted_confidence = confidence
            adjustment_reason = f"Giữ nguyên confidence - độ chính xác trung bình ({overall_accuracy:.1%})"
        
        # Điều chỉnh prediction dựa trên model type performance
        model_type = current_prediction.get('model_type', 'unknown')
        if model_type in accuracy_stats.get('model_type_stats', {}):
            model_confidence = accuracy_stats['model_type_stats'][model_type]['confidence']['mean']
            if model_confidence < 0.5:
                # Model này có hiệu suất kém, giảm confidence thêm
                adjusted_confidence *= 0.9
                adjustment_reason += f", giảm thêm do model {model_type} kém ({model_confidence:.1%})"
        
        # Cập nhật prediction
        adjusted_prediction = current_prediction.copy()
        adjusted_prediction['confidence'] = adjusted_confidence
        adjusted_prediction['original_confidence'] = confidence
        adjusted_prediction['adjustment_reason'] = adjustment_reason
        adjusted_prediction['historical_accuracy'] = overall_accuracy
        
        logger.info(f"🔧 Điều chỉnh thuật toán ML cho {symbol} ({timeframe}): {confidence:.1%} → {adjusted_confidence:.1%}")
        logger.info(f"📊 Lý do: {adjustment_reason}")
        
        return adjusted_prediction
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi điều chỉnh thuật toán ML: {e}")
        return current_prediction

def main():
    logger.info("Bắt đầu phân tích xu hướng ngắn hạn với ML và Convergence Analysis...")
    
    # Đảm bảo các thư mục cần thiết
    ensure_prediction_data_dir()
    ensure_ml_directories()
    
    # Hiển thị thông tin ML features
    display_ml_features_info()
    
    # Kiểm tra trạng thái ML training và dữ liệu
    ml_status = get_ml_training_status()
    if ml_status:
        print(f"\n📊 ML Training Status:")
        print(f"✅ Models trained: {len(ml_status['models_trained'])}")
        print(f"❌ Models missing: {len(ml_status['models_missing'])}")
        print(f"📁 Data files: {len(ml_status['data_files'])}")
    
    # Hiển thị tóm tắt dữ liệu
    display_data_update_summary()
    
    # Train ML models với incremental data update
    logger.info("🤖 Bắt đầu train ML models với incremental data update...")
    symbols_to_train = ['BTC/USDT', 'ETH/USDT']
    timeframes_to_train = ML_TIMEFRAMES
    
    # Hiển thị thống kê dữ liệu trước khi train
    logger.info("📊 Thống kê dữ liệu hiện tại:")
    for symbol in symbols_to_train:
        for timeframe in timeframes_to_train:
            stats = get_data_statistics(symbol, timeframe)
            if stats:
                logger.info(f"  {symbol} ({timeframe}): {stats['total_candles']} candles, {stats['file_size_mb']:.2f}MB, {stats['date_range']['start']} - {stats['date_range']['end']}")
            else:
                logger.info(f"  {symbol} ({timeframe}): Chưa có dữ liệu")
    
    for symbol in symbols_to_train:
        for timeframe in timeframes_to_train:
            logger.info(f"🔄 Training ML models cho {symbol} ({timeframe})...")
            try:
                # Kiểm tra xem có cần force full update không
                is_fresh, freshness_msg = check_data_freshness(symbol, timeframe, max_age_hours=48)
                force_full = not is_fresh
                
                if force_full:
                    logger.info(f"🔄 Force full update cho {symbol} ({timeframe}) - {freshness_msg}")
                
                train_ml_models(symbol, timeframe, force_full_update=force_full)
                logger.info(f"✅ Đã train thành công cho {symbol} ({timeframe})")
            except Exception as e:
                logger.error(f"❌ Lỗi train {symbol} ({timeframe}): {e}")
    
    # Phân tích các symbols
    symbols = get_usdt_symbols()
    logger.info(f"Đã chọn {len(symbols)} tài sản: {symbols}")
    
    results = []
    for symbol in symbols:
        result = analyze_coin(symbol)
        if result:
            results.append(result)
            logger.info(f"✅ Đã phân tích {symbol} thành công")
        else:
            logger.warning(f"⚠️ Không thể phân tích {symbol}")



    # Hiển thị thống kê độ chính xác ML
    logger.info("📊 Thống kê độ chính xác ML:")
    for symbol in ['BTC_USDT', 'ETH_USDT']:
        for timeframe in ML_TIMEFRAMES:
            accuracy_stats = get_prediction_accuracy_stats(symbol, timeframe, days_back=7)
            if accuracy_stats:
                verified = accuracy_stats['verified_predictions']
                failed = accuracy_stats['failed_predictions']
                expired = accuracy_stats['expired_predictions']
                total = accuracy_stats['total_predictions']
                accuracy = accuracy_stats['overall_accuracy']
                
                logger.info(f"  {symbol} ({timeframe}): {accuracy:.1%} ({verified} đúng, {failed} sai, {expired} hết hạn) trong 7 ngày qua")
            else:
                logger.info(f"  {symbol} ({timeframe}): Chưa có dữ liệu độ chính xác")
    
    # Hiển thị thống kê độ chính xác tổng thể nếu có
    accuracy_data = get_prediction_accuracy_data()
    if accuracy_data and accuracy_data.get('overall', {}).get('total_predictions', 0) > 0:
        overall = accuracy_data['overall']
        logger.info(f"📈 Thống kê độ chính xác tổng thể: {overall['accuracy']:.1%} ({overall['accurate_predictions']}/{overall['total_predictions']})")
    
    # Test Telegram trước
    logger.info("🧪 TESTING TELEGRAM CONNECTION...")
    test_success = send_telegram_message("🤖 Bot test message - Chỉ gửi báo cáo BTC!")
    if test_success:
        logger.info("✅ Telegram test thành công!")
    else:
        logger.error("❌ Telegram test thất bại!")
    
    # Gửi báo cáo Telegram chỉ cho BTC
    logger.info("🔍 Lọc kết quả để chỉ gửi báo cáo BTC...")
    btc_results = [result for result in results if result['symbol'] == 'BTC/USDT']
    eth_results = [result for result in results if result['symbol'] == 'ETH/USDT']
    
    logger.info(f"📊 Tổng kết quả: {len(results)} (BTC: {len(btc_results)}, ETH: {len(eth_results)})")
    
    if btc_results:
        logger.info("📱 CHỈ GỬI BÁO CÁO BTC - KHÔNG GỬI ETH!")
        report = format_analysis_report(btc_results)
        success = send_telegram_message(report)
        if success:
            logger.info("✅ Đã gửi báo cáo Telegram cho BTC thành công!")
        else:
            logger.error("❌ Lỗi gửi báo cáo Telegram cho BTC")
    else:
        logger.info("📊 Không có kết quả phân tích BTC để gửi")
        # Gửi thông báo không có tín hiệu cho BTC
        no_signal_report = "🤖 <b>BÁO CÁO PHÂN TÍCH BTC</b>\n\n📊 Không có tín hiệu mạnh nào được phát hiện cho BTC trong thị trường hiện tại.\n\n💡 Điều này có thể do:\n• Thị trường đang sideway/consolidation\n• Các chỉ số chưa đạt ngưỡng tín hiệu\n• Cần chờ thêm thời gian để có tín hiệu rõ ràng\n\n⏰ Thời gian: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        send_telegram_message(no_signal_report)
    
    # Log thông tin về ETH (không gửi báo cáo)
    if eth_results:
        logger.info("📊 Đã phân tích ETH nhưng KHÔNG GỬI BÁO CÁO (theo yêu cầu)")
    else:
        logger.info("📊 Không có kết quả phân tích ETH")
    
    # Kiểm tra tình trạng dữ liệu lịch sử (không xóa, chỉ kiểm tra)
    logger.info("🧹 Kiểm tra tình trạng dữ liệu lịch sử...")
    cleanup_old_data_files()
    
    logger.info("🏁 Hoàn thành phân tích!")

if __name__ == "__main__":
    main()
