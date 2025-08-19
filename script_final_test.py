#!/usr/bin/env python3
"""
Script cuối cùng - chỉ phân tích và gửi Telegram
Không có scheduler, không có infinite loop
"""

# Import tất cả functions từ script.py
exec(open('script.py').read())

def main_final_test():
    """Main function cuối cùng"""
    logger.info("🚀 Bắt đầu phân tích cuối cùng...")
    
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
    
    # Train ML models trực tiếp (không dùng scheduler)
    logger.info("🤖 Bắt đầu train ML models...")
    
    # Train cho BTC và ETH
    symbols_to_train = ['BTC/USDT', 'ETH/USDT']
    timeframes_to_train = ['1h', '4h', '1d']
    
    for symbol in symbols_to_train:
        for timeframe in timeframes_to_train:
            logger.info(f"🔄 Training ML models cho {symbol} ({timeframe})...")
            try:
                train_ml_models(symbol, timeframe)
                logger.info(f"✅ Đã train thành công cho {symbol} ({timeframe})")
            except Exception as e:
                logger.error(f"❌ Lỗi train {symbol} ({timeframe}): {e}")
    
    # Lấy symbols để phân tích
    symbols = get_usdt_symbols()
    logger.info(f"Đã chọn {len(symbols)} tài sản: {symbols}")
    
    # Phân tích từng symbol
    results = []
    for symbol in symbols:
        logger.info(f"🔍 Đang phân tích {symbol}...")
        try:
            result = analyze_coin(symbol)
            if result:
                results.append(result)
                logger.info(f"✅ Đã phân tích {symbol} thành công")
            else:
                logger.warning(f"⚠️ Không thể phân tích {symbol}")
        except Exception as e:
            logger.error(f"❌ Lỗi phân tích {symbol}: {e}")
    
    # Hiển thị thống kê độ chính xác nếu có
    accuracy_data = get_prediction_accuracy_data()
    if accuracy_data and accuracy_data.get('overall', {}).get('total_predictions', 0) > 0:
        overall = accuracy_data['overall']
        logger.info(f"📈 Thống kê độ chính xác: {overall['accuracy']:.1%} ({overall['accurate_predictions']}/{overall['total_predictions']})")
    
    logger.info(f"🎉 Phân tích hoàn thành! Đã phân tích {len(results)} symbols")
    logger.info("📱 Kết quả đã được gửi về Telegram")
    
    return results

if __name__ == "__main__":
    main_final_test()
