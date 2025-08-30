# Incremental Data Update System

## Tổng quan

Hệ thống incremental data update cho phép bot cập nhật dữ liệu ML một cách thông minh, chỉ tải dữ liệu mới thay vì tải lại toàn bộ dữ liệu mỗi lần chạy. Điều này giúp:

- **Tiết kiệm thời gian**: Chỉ tải dữ liệu mới từ lần chạy cuối
- **Tiết kiệm băng thông**: Giảm lượng dữ liệu cần tải từ API
- **Tăng hiệu suất**: Dữ liệu được cache locally và merge thông minh
- **Giữ kích thước repository hợp lý**: Tự động dọn dẹp dữ liệu cũ

## Các tính năng chính

### 1. Incremental Data Loading
- **`load_and_update_historical_data()`**: Load dữ liệu hiện có và merge với dữ liệu mới
- **`fetch_incremental_data()`**: Lấy dữ liệu mới từ timestamp cuối cùng
- **`merge_historical_data()`**: Merge dữ liệu cũ và mới, loại bỏ duplicates

### 2. Data Freshness Checking
- **`check_data_freshness()`**: Kiểm tra độ mới của dữ liệu (mặc định 24h)
- **`get_data_statistics()`**: Lấy thống kê chi tiết về dữ liệu
- **`display_data_update_summary()`**: Hiển thị tóm tắt tình trạng dữ liệu

### 3. Smart Data Management
- **`cleanup_old_data_files()`**: Tự động dọn dẹp file cũ (giữ tối đa 10,000 candles/file)
- **`get_latest_timestamp_from_data()`**: Lấy timestamp mới nhất từ dữ liệu

## Cách hoạt động

### Lần chạy đầu tiên
```python
# Force full update - tải toàn bộ dữ liệu
data = load_and_update_historical_data(symbol, timeframe, force_full_update=True)
```

### Các lần chạy tiếp theo
```python
# Incremental update - chỉ tải dữ liệu mới
data = load_and_update_historical_data(symbol, timeframe, force_full_update=False)
```

### Logic merge dữ liệu
1. Load dữ liệu hiện có từ file CSV
2. Lấy timestamp mới nhất
3. Fetch dữ liệu mới từ API từ timestamp đó
4. Merge và loại bỏ duplicates
5. Lưu lại file đã merge

## Cấu hình

### Thời gian cache dữ liệu
```python
# Trong script.py
max_age_hours = 24  # Dữ liệu được coi là "mới" trong 24h
```

### Giới hạn kích thước file
```python
max_candles_per_file = 10000  # Tối đa 10,000 candles/file
ML_HISTORICAL_CANDLES = 5000  # Số candles mặc định cho ML training
```

## Monitoring và Logging

### Thống kê dữ liệu
```bash
📊 TÓM TẮT CẬP NHẬT DỮ LIỆU:
==================================================
✅ BTC/USDT (1h): 5000 candles, 2.34MB
   📅 2024-01-01 00:00 → 2024-01-15 12:00
   🕒 Dữ liệu mới (2.5 giờ)
✅ ETH/USDT (4h): 3000 candles, 1.45MB
   📅 2024-01-01 00:00 → 2024-01-15 12:00
   🕒 Dữ liệu mới (1.2 giờ)
==================================================
📈 Tổng cộng: 14 files, 35,000 candles, 15.67MB
```

### Log messages
```
📁 Loaded 5000 existing candles for BTC/USDT (1h)
🔄 Merged data: 5000 existing + 24 new = 5024 total
💾 Saved 5024 candles to ml_data/BTC_USDT_1h_historical.csv
```

## Testing

Chạy test script để kiểm tra tính năng:
```bash
python test_incremental_data.py
```

Test script sẽ kiểm tra:
- Force full update
- Incremental update
- Data consistency
- File cleanup
- Statistics generation

## Lợi ích

### So với hệ thống cũ
| Tiêu chí | Hệ thống cũ | Hệ thống mới |
|----------|-------------|--------------|
| Thời gian tải dữ liệu | 30-60 giây | 5-15 giây |
| Lượng dữ liệu tải | 5000 candles | 24-100 candles |
| Băng thông sử dụng | Cao | Thấp |
| Hiệu suất | Chậm | Nhanh |
| Độ tin cậy | Thấp (phụ thuộc API) | Cao (có cache) |

### Tối ưu hóa cho GitHub Actions
- **Giảm thời gian chạy**: Từ 5-10 phút xuống 2-3 phút
- **Giảm lỗi API**: Ít request hơn = ít lỗi hơn
- **Tiết kiệm quota**: Sử dụng ít API calls hơn
- **Dữ liệu liên tục**: Không bị mất dữ liệu giữa các lần chạy

## Troubleshooting

### Dữ liệu không được cập nhật
```python
# Force full update
data = load_and_update_historical_data(symbol, timeframe, force_full_update=True)
```

### File dữ liệu bị corrupt
```python
# Xóa file và tải lại
import os
os.remove(f"ml_data/{symbol}_{timeframe}_historical.csv")
data = load_and_update_historical_data(symbol, timeframe, force_full_update=True)
```

### Kiểm tra tình trạng dữ liệu
```python
stats = get_data_statistics(symbol, timeframe)
is_fresh, msg = check_data_freshness(symbol, timeframe)
print(f"Status: {msg}")
```

## Tương lai

### Cải tiến có thể thêm
- **Compression**: Nén dữ liệu để giảm kích thước file
- **Backup**: Tự động backup dữ liệu quan trọng
- **Validation**: Kiểm tra tính hợp lệ của dữ liệu
- **Parallel processing**: Tải dữ liệu song song cho nhiều symbols
- **Cloud storage**: Lưu trữ dữ liệu trên cloud thay vì local

### Monitoring nâng cao
- **Data quality metrics**: Đo lường chất lượng dữ liệu
- **Performance metrics**: Theo dõi hiệu suất tải dữ liệu
- **Alert system**: Cảnh báo khi có vấn đề với dữ liệu
