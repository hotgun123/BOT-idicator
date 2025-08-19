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

# Khởi tạo kết nối với Binance mainnet (spot)
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
    }
})

# Khởi tạo kết nối với Exness cho hàng hóa (đã loại bỏ)
exness_exchange = None

# Cấu hình
# Chỉ phân tích crypto; tạm thời bỏ vàng và dầu do nguồn dữ liệu không ổn định
SYMBOLS = ['BTC/USDT', 'ETH/USDT']  # Bỏ BNB theo yêu cầu của user
TIMEFRAMES = ['1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
ML_TIMEFRAMES = ['1h', '2h', '4h', '6h', '8h', '12h', '1d']  # Timeframes cho ML training
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
    """Trả về danh sách cặp giao dịch cố định bao gồm crypto, vàng và dầu"""
    return SYMBOLS

def ensure_prediction_data_dir():
    """Đảm bảo thư mục dữ liệu dự đoán tồn tại"""
    Path(PREDICTION_DATA_DIR).mkdir(exist_ok=True)

def ensure_ml_directories():
    """Đảm bảo các thư mục ML tồn tại"""
    Path(ML_MODELS_DIR).mkdir(exist_ok=True)
    Path(ML_DATA_DIR).mkdir(exist_ok=True)

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
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
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

def train_ml_models(symbol, timeframe):
    """Train các mô hình Machine Learning với dữ liệu lịch sử"""
    try:
        ensure_ml_directories()
        
        # Lấy dữ liệu lịch sử (5000 candles)
        data = load_or_fetch_historical_data(symbol, timeframe)
        if data is None:
            logger.warning(f"⚠️ Không thể lấy dữ liệu lịch sử cho {symbol} ({timeframe})")
            return None
            
        logger.info(f"📊 Dữ liệu {symbol} ({timeframe}): {len(data['close'])} candles")
        
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
        elif prediction_class == 0 and confidence > ML_CONFIDENCE_THRESHOLD:
            signal = 'Short'
        else:
            signal = 'Hold'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'probability': prediction_proba,
            'model_name': best_model_name,
            'model_performance': performance[best_model_name]
        }
        
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
    """Lấy giá hiện tại cho việc cập nhật dự đoán"""
    try:
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

# Đã loại bỏ tất cả các hàm liên quan đến hàng hóa (vàng, dầu)

def fetch_ohlcv(symbol, timeframe, limit):
    """Lấy dữ liệu OHLCV cho crypto"""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            # Xử lý cho crypto
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

    # === 1. TREND INDICATORS (3 chỉ số cốt lõi) ===
    ema20 = ta.trend.ema_indicator(close, window=20)              # Trend ngắn hạn
    ema50 = ta.trend.ema_indicator(close, window=50)              # Trend trung hạn  
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
    support, resistance = find_support_resistance(high, low, current_price)

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
    
    # === 8. PHÂN TÍCH DIVERGENCE/CONVERGENCE - TRỌNG SỐ CAO ===
    divergences = analyze_all_divergences(close, rsi, macd_line, volume)
    divergence_consensus = calculate_divergence_consensus(divergences)
    
    # === 9. TÍNH TOÁN TÍN HIỆU CƠ BẢN ===
    
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
    ma_distance = abs(get_last(ema20) - get_last(ema50)) / get_last(ema50)
    if get_last(ema20) > get_last(ema50) and ma_distance > 0.01:
        ma_signal = 'Long'
    elif get_last(ema20) < get_last(ema50) and ma_distance > 0.01:
        ma_signal = 'Short'

    # ADX Signal
    adx_signal = 'Hold'
    if get_last(adx) > 25:
        if get_last(close) > get_last(ema20):
            adx_signal = 'Long'
        elif get_last(close) < get_last(ema20):
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
        candlestick_signal, price_pattern_signal,
        smc_signals['order_block_signal'], smc_signals['fvg_signal'], 
        smc_signals['liquidity_signal'], smc_signals['mitigation_signal']
    ]

    # === 12. XỬ LÝ DIVERGENCE VỚI TRỌNG SỐ CAO ===
    divergence_signal = divergence_consensus['signal']
    divergence_strength = divergence_consensus['strength']
    divergence_count = divergence_consensus['count']
    
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

    # === 14. TÍNH TOÁN CONSENSUS CUỐI CÙNG ===
    all_signals = final_signals + extra_signals
    
    long_count = all_signals.count('Long')
    short_count = all_signals.count('Short')
    hold_count = all_signals.count('Hold')
    
    total_signals = len(all_signals)
    
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
    entry_points = calculate_entry_points(current_price, high, low, close, rsi, bb_upper, bb_lower, ema50, pivot_points, support, resistance)

    # === 15. MACHINE LEARNING PREDICTION ===
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

    # === 16. PHÂN TÍCH HỘI TỤ (CONVERGENCE ANALYSIS) ===
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

    # === 17. TÍNH TOÁN CONSENSUS CUỐI CÙNG (CẬP NHẬT) ===
    all_signals = final_signals + extra_signals
    
    long_count = all_signals.count('Long')
    short_count = all_signals.count('Short')
    hold_count = all_signals.count('Hold')
    
    total_signals = len(all_signals)
    
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

    # === 18. TRẢ VỀ KẾT QUẢ TỐI ƯU ===
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
            'ema20': get_last(ema20),
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
        }
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
    """Phân tích xu hướng ngắn hạn cho một coin"""
    try:
        logger.info(f"🔍 Bắt đầu phân tích {symbol}...")
        
        # Lấy giá hiện tại cho crypto
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
    logger.info(f"🔍 DEBUG: Bắt đầu gửi Telegram message")
    logger.info(f"🔍 DEBUG: TELEGRAM_BOT_TOKEN = {TELEGRAM_BOT_TOKEN[:10]}...")
    logger.info(f"🔍 DEBUG: TELEGRAM_CHAT_ID = {TELEGRAM_CHAT_ID}")
    
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
        logger.info(f"🔍 DEBUG: Gửi request đến {url}")
        logger.info(f"🔍 DEBUG: Data = {data}")
        
        response = requests.post(url, data=data, timeout=10)
        logger.info(f"🔍 DEBUG: Response status = {response.status_code}")
        logger.info(f"🔍 DEBUG: Response text = {response.text}")
        
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
    """Định dạng báo cáo phân tích cho một đồng coin, vàng hoặc dầu cụ thể - Tối ưu cho tín hiệu mạnh"""
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
        report += f"💡 Tín hiệu từ timeframe có đồng thuận cao nhất\n\n"
        
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
    report += f"💰 Tài sản: Crypto, Vàng, Dầu\n"
    report += f"🎯 <b>12 CHỈ SỐ CỐT LÕI + ML + CONVERGENCE ANALYSIS</b>\n\n"
    
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
            
            # Thêm thông tin chi tiết cho từng timeframe
            for analysis in valid_timeframes:
                timeframe = analysis['timeframe']
                report += f"\n📊 <b>Timeframe {timeframe}:</b>\n"
                
                # === 1. MACHINE LEARNING PREDICTION ===
                ml_prediction = analysis.get('ml_prediction')
                if ml_prediction and ml_prediction.get('confidence', 0) > ML_CONFIDENCE_THRESHOLD:
                    report += f"🤖 <b>MACHINE LEARNING PREDICTION:</b>\n"
                    report += f"  • Model: {ml_prediction['model_name']}\n"
                    report += f"  • Tín hiệu: {ml_prediction['signal']}\n"
                    report += f"  • Confidence: {ml_prediction['confidence']:.3f}\n"
                    report += f"  • Accuracy: {ml_prediction['model_performance']['accuracy']:.3f}\n"
                    report += f"  • CV Score: {ml_prediction['model_performance']['cv_mean']:.3f}±{ml_prediction['model_performance']['cv_std']:.3f}\n\n"

                # === 2. CONVERGENCE ANALYSIS ===
                convergence_analysis = analysis.get('convergence_analysis')
                if convergence_analysis and convergence_analysis.get('overall_convergence', 0) > CONVERGENCE_THRESHOLD:
                    report += f"🎯 <b>CONVERGENCE ANALYSIS:</b>\n"
                    report += f"  • Overall Convergence: {convergence_analysis['overall_convergence']:.3f}\n"
                    report += f"  • Strength: {convergence_analysis['strength']:.3f}\n"
                    report += f"  • Signals: {len(convergence_analysis['signals'])}\n"
                    
                    for signal in convergence_analysis['signals']:
                        report += f"  • {signal['period']} periods: {signal['signal']} (Strength: {signal['strength']:.3f})\n"
                    report += "\n"

                # === 3. DIVERGENCE/CONVERGENCE - ƯU TIÊN CAO NHẤT ===
                divergence_consensus = analysis.get('divergence_consensus', {})
                if divergence_consensus.get('signal') != 'Hold' and divergence_consensus.get('strength', 0) > 0.2:
                    strength_emoji = "🔥" if divergence_consensus['strength'] > 0.5 else "⚡"
                    report += f"{strength_emoji} <b>DIVERGENCE/CONVERGENCE MẠNH:</b>\n"
                    report += f"  • Tín hiệu: {divergence_consensus['signal']}\n"
                    report += f"  • Độ mạnh: {divergence_consensus['strength']:.2f}\n"
                    report += f"  • Số lượng: {divergence_consensus['count']}\n"
                    
                    # Hiển thị chi tiết divergence
                    divergences = analysis.get('divergences', {})
                    for div_type, div_info in divergences.items():
                        if div_info.get('type') != 'None':
                            report += f"  • {div_type}: {div_info['type']} ({div_info['strength']:.2f})\n"
                    report += "\n"
                
                # === 2. CHỈ SỐ CỐT LÕI ===
                signals = analysis.get('signals', {})
                indicators = analysis.get('indicators', {})
                
                # Trend Indicators
                report += f"📈 <b>TREND:</b>\n"
                report += f"  • MA: {signals.get('ma', 'Hold')} (EMA20: {indicators.get('ema20', 0):.4f})\n"
                report += f"  • ADX: {signals.get('adx', 'Hold')} ({indicators.get('adx', 0):.1f})\n"
                
                # Momentum Indicators
                report += f"📊 <b>MOMENTUM:</b>\n"
                report += f"  • RSI: {signals.get('rsi', 'Hold')} ({indicators.get('rsi', 0):.1f})\n"
                report += f"  • Stochastic: {signals.get('stoch', 'Hold')} (K: {indicators.get('stoch_k', 0):.1f})\n"
                report += f"  • MACD: {signals.get('macd', 'Hold')} ({indicators.get('macd_line', 0):.4f})\n"
                
                # Volatility Indicators
                report += f"📉 <b>VOLATILITY:</b>\n"
                report += f"  • Bollinger Bands: {signals.get('bb', 'Hold')}\n"
                report += f"  • ATR: {signals.get('atr', 'Hold')} ({indicators.get('atr', 0):.4f})\n"
                
                # Volume Indicators
                report += f"💰 <b>VOLUME:</b>\n"
                report += f"  • OBV: {signals.get('obv', 'Hold')}\n"
                report += f"  • VWAP: {signals.get('vwap', 'Hold')} ({indicators.get('vwap', 0):.4f})\n"
                
                # Support/Resistance
                report += f"🎯 <b>SUPPORT/RESISTANCE:</b>\n"
                report += f"  • Pivot: {signals.get('pivot', 'Hold')}\n"
                report += f"  • Support: {indicators.get('support', 0):.4f}\n"
                report += f"  • Resistance: {indicators.get('resistance', 0):.4f}\n"
                
                # Patterns
                if analysis.get('price_pattern') != 'None':
                    report += f"📊 <b>MÔ HÌNH GIÁ:</b> {analysis['price_pattern']}\n"
                
                if analysis.get('candlestick_patterns'):
                    report += f"🕯️ <b>MÔ HÌNH NẾN:</b> {', '.join(analysis['candlestick_patterns'])}\n"
                
                # Smart Money Concepts
                smc_signals = analysis.get('smc_signals', {})
                if any(signal != 'Hold' for signal in smc_signals.values()):
                    report += f"🧠 <b>SMART MONEY CONCEPTS:</b>\n"
                    for smc_type, smc_signal in smc_signals.items():
                        if smc_signal != 'Hold':
                            report += f"  • {smc_type}: {smc_signal}\n"
                
                # Commodity Signals (cho vàng và dầu)
                commodity_signals = analysis.get('commodity_signals', {})
                if commodity_signals:
                    report += f"🏆 <b>CHỈ SỐ HÀNG HÓA:</b>\n"
                    for comm_type, comm_signal in commodity_signals.items():
                        if comm_signal != 'Hold':
                            report += f"  • {comm_type}: {comm_signal}\n"
                
                # Signal Counts
                signal_counts = analysis.get('signal_counts', {})
                if signal_counts:
                    report += f"📊 <b>THỐNG KÊ TÍN HIỆU:</b>\n"
                    report += f"  • Long: {signal_counts.get('long', 0)}\n"
                    report += f"  • Short: {signal_counts.get('short', 0)}\n"
                    report += f"  • Hold: {signal_counts.get('hold', 0)}\n"
                    report += f"  • Tổng: {signal_counts.get('total', 0)}\n"
                
                # Entry Points
                if 'entry_points' in analysis:
                    entry = analysis['entry_points']
                    report += f"\n🎯 <b>ĐIỂM ENTRY HỢP LÝ:</b>\n"
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
    
    # Helper function để lấy giá trị cuối cùng
    def get_last(series):
        return series.iloc[-1] if hasattr(series, 'iloc') else series[-1]
    
    # 1. Phân tích xu hướng hiện tại
    trend = 'neutral'
    if current_price > get_last(ema50):
        trend = 'bullish'
    else:
        trend = 'bearish'
    
    # 2. Tính các mức entry cho Long
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
        conservative_entry = max(resistance, get_last(bb_upper), pivot_points['r1'])
        entry_points['conservative'] = conservative_entry
        
        # Entry tích cực - Vào ngay khi có tín hiệu
        aggressive_entry = current_price * 1.005  # Vào cao hơn giá hiện tại 0.5%
        entry_points['aggressive'] = aggressive_entry
        
        # Stop Loss - Dựa trên mức kháng cự mạnh (r2) để tạo R/R tốt hơn
        # Sử dụng r2 thay vì r1 để SL gần entry hơn
        stop_loss = max(resistance * 1.002, get_last(bb_upper) * 1.001, pivot_points['r2'] * 1.001)
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
    if get_last(rsi) < 15:  # Từ 20 -> 15
        entry_points['analysis'].append(f"  • RSI quá bán ({get_last(rsi):.1f}) → Ưu tiên entry bảo thủ")
    elif get_last(rsi) > 85:  # Từ 80 -> 85
        entry_points['analysis'].append(f"  • RSI quá mua ({get_last(rsi):.1f}) → Ưu tiên entry bảo thủ")
    else:
        entry_points['analysis'].append(f"  • RSI trung tính ({get_last(rsi):.1f}) → Có thể entry tích cực")
    
    # 5. Phân tích Bollinger Bands
    if current_price < get_last(bb_lower):
        entry_points['analysis'].append(f"  • Giá dưới BB Lower → Cơ hội entry tốt cho Long")
    elif current_price > get_last(bb_upper):
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
    """Phân tích tất cả các loại divergence"""
    divergences = []
    
    # RSI Divergence
    rsi_div = analyze_rsi_divergence(close_prices, rsi_values)
    if rsi_div:
        divergences.append(rsi_div)
    
    # MACD Divergence
    macd_div = analyze_macd_divergence(close_prices, macd_values)
    if macd_div:
        divergences.append(macd_div)
    
    # Price-Volume Divergence
    volume_div = analyze_price_volume_divergence(close_prices, volume_data)
    if volume_div:
        divergences.append(volume_div)
    
    return divergences

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
    """Lấy dữ liệu lịch sử cho ML training từ Binance API"""
    try:
        if limit is None:
            limit = ML_HISTORICAL_CANDLES
            
        logger.info(f"📊 Đang lấy dữ liệu lịch sử cho {symbol} ({timeframe}) - {limit} candles...")
        
        # Lấy dữ liệu từ Binance
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv or len(ohlcv) < ML_MIN_SAMPLES:
            logger.warning(f"⚠️ Không đủ dữ liệu lịch sử cho {symbol} ({timeframe}): {len(ohlcv) if ohlcv else 0} candles")
            return None
        
        # Chuyển đổi thành DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Lưu dữ liệu gốc (thay thế ký tự / bằng _)
        safe_symbol = symbol.replace('/', '_')
        data_file = os.path.join(ML_DATA_DIR, f"{safe_symbol}_{timeframe}_historical.csv")
        df.to_csv(data_file)
        
        logger.info(f"✅ Đã lấy và lưu {len(df)} candles lịch sử cho {symbol} ({timeframe})")
        
        return {
            'open': df['open'].values,
            'high': df['high'].values,
            'low': df['low'].values,
            'close': df['close'].values,
            'volume': df['volume'].values,
            'timestamp': df.index.values
        }
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy dữ liệu lịch sử cho {symbol} ({timeframe}): {e}")
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
                logger.info(f"📂 Loading dữ liệu lịch sử từ file cho {symbol} ({timeframe})...")
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
        'Support/Resistance (2)': [
            'Support Distance - Distance to Support Level',
            'Resistance Distance - Distance to Resistance Level'
        ],
        'Market Structure (2)': [
            'Trend Strength - Higher Highs vs Lower Lows',
            'Price Acceleration - Rate of Price Change'
        ],
        'Price Patterns (2)': [
            'Hammer - Hammer Candlestick Pattern',
            'Doji - Doji Candlestick Pattern'
        ]
    }
    
    print("\n🤖 MACHINE LEARNING FEATURES (35+ Features)")
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

def get_ml_training_status():
    """Kiểm tra trạng thái training ML models"""
    try:
        ensure_ml_directories()
        
        status = {
            'models_trained': [],
            'models_missing': [],
            'last_training': None,
            'data_files': []
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
        
        # Kiểm tra data files
        data_files = [f for f in os.listdir(ML_DATA_DIR) if f.endswith('_historical.csv')]
        status['data_files'] = data_files
        
        return status
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi kiểm tra trạng thái ML: {e}")
        return None

def main():
    logger.info("Bắt đầu phân tích xu hướng ngắn hạn với ML và Convergence Analysis...")
    
    # Đảm bảo các thư mục cần thiết
    ensure_prediction_data_dir()
    ensure_ml_directories()
    
    # Hiển thị thông tin ML features
    display_ml_features_info()
    
    # Kiểm tra trạng thái ML training
    ml_status = get_ml_training_status()
    if ml_status:
        print(f"\n📊 ML Training Status:")
        print(f"✅ Models trained: {len(ml_status['models_trained'])}")
        print(f"❌ Models missing: {len(ml_status['models_missing'])}")
        print(f"📁 Data files: {len(ml_status['data_files'])}")
    
    # Train ML models một lần (không có scheduler)
    logger.info("🤖 Bắt đầu train ML models...")
    symbols_to_train = ['BTC/USDT', 'ETH/USDT']
    timeframes_to_train = ML_TIMEFRAMES
    
    for symbol in symbols_to_train:
        for timeframe in timeframes_to_train:
            logger.info(f"🔄 Training ML models cho {symbol} ({timeframe})...")
            try:
                train_ml_models(symbol, timeframe)
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

    # Hiển thị thống kê độ chính xác nếu có
    accuracy_data = get_prediction_accuracy_data()
    if accuracy_data and accuracy_data.get('overall', {}).get('total_predictions', 0) > 0:
        overall = accuracy_data['overall']
        logger.info(f"📈 Thống kê độ chính xác: {overall['accuracy']:.1%} ({overall['accurate_predictions']}/{overall['total_predictions']})")
    
    # Gửi báo cáo Telegram
    logger.info(f"🔍 DEBUG: Có {len(results)} kết quả để gửi")
    if results:
        report = format_analysis_report(results)
        logger.info(f"🔍 DEBUG: Report length = {len(report)} characters")
        logger.info(f"🔍 DEBUG: Report preview = {report[:200]}...")
        success = send_telegram_message(report)
        if success:
            logger.info("📱 Đã gửi báo cáo Telegram thành công!")
        else:
            logger.error("❌ Lỗi gửi báo cáo Telegram")
    else:
        logger.info("📊 Không có kết quả phân tích để gửi")
    
    logger.info("🏁 Hoàn thành phân tích!")

if __name__ == "__main__":
    main()
