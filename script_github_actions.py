#!/usr/bin/env python3
"""
Script tối ưu cho GitHub Actions - chạy một lần và thoát
Không có infinite loop, chỉ phân tích và gửi kết quả
"""

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
from dotenv import load_dotenv
import ta

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

# Import tất cả functions từ script.py
exec(open('script.py').read())

def main_github_actions():
    """Main function tối ưu cho GitHub Actions"""
    logger.info("🚀 Bắt đầu phân tích cho GitHub Actions...")
    
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
    
    # Lấy symbols để phân tích
    symbols = get_usdt_symbols()
    logger.info(f"Đã chọn {len(symbols)} tài sản: {symbols}")
    
    # Phân tích từng symbol
    results = []
    for symbol in symbols:
        logger.info(f"🔍 Đang phân tích {symbol}...")
        result = analyze_coin(symbol)
        if result:
            results.append(result)
            logger.info(f"✅ Đã phân tích {symbol} thành công")
        else:
            logger.warning(f"⚠️ Không thể phân tích {symbol}")
    
    # Hiển thị thống kê độ chính xác nếu có
    accuracy_data = get_prediction_accuracy_data()
    if accuracy_data and accuracy_data.get('overall', {}).get('total_predictions', 0) > 0:
        overall = accuracy_data['overall']
        logger.info(f"📈 Thống kê độ chính xác: {overall['accuracy']:.1%} ({overall['accurate_predictions']}/{overall['total_predictions']})")
    
    logger.info(f"🎉 Phân tích hoàn thành! Đã phân tích {len(results)} symbols")
    logger.info("📱 Kết quả đã được gửi về Telegram")
    
    return results

if __name__ == "__main__":
    main_github_actions()
