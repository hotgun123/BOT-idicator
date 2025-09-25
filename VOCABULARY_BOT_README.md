# 🤖 Vocabulary Bot - Bot học từ vựng tiếng Anh

## 📋 Mô tả
Bot học từ vựng tiếng Anh độc lập, gửi từ vựng B1-C1 qua Telegram riêng biệt.

## 🔧 Cấu hình
- **Bot Token**: `8085678497:AAFPzATgWObijU3o1TJyUY9ukfPPGR06Tps`
- **Chat ID**: `1866335373`
- **Lịch gửi**: Mỗi 3 giờ (6h, 9h, 12h, 15h, 18h, 21h, 0h, 3h UTC)

## 📁 Files
- `vocabulary_bot.py` - Bot chính
- `vocabulary_learning.py` - Module từ vựng
- `vocabulary_config.py` - Cấu hình bot
- `VOCABULARY_BOT_README.md` - Hướng dẫn này

## 🚀 Cách sử dụng

### 1. Chạy bot từ vựng riêng:
```bash
python vocabulary_bot.py
```

### 2. Test bot:
```bash
python -c "from vocabulary_learning import send_vocabulary_lesson; send_vocabulary_lesson()"
```

### 3. Xem thống kê từ vựng:
```bash
python -c "from vocabulary_learning import get_vocabulary_stats; print(get_vocabulary_stats())"
```

## ⚙️ Cấu hình lịch gửi
Chỉnh sửa `vocabulary_config.py`:
```python
VOCABULARY_SCHEDULE = {
    'enabled': True,
    'interval_hours': 8,
    'times': [6, 14, 22],  # Giờ gửi (UTC) - 3 lần/ngày
    'words_per_lesson': 5  # 5 từ mỗi bài
}
```

## 📊 Tính năng
- ✅ **78 từ vựng** A2-B2
- ✅ **5 từ mỗi bài** gửi
- ✅ **Nghĩa tiếng Việt** + câu ví dụ
- ✅ **Lịch tự động** 3 lần/ngày
- ✅ **Bot Telegram riêng** biệt
- ✅ **Logging** chi tiết

## 🔄 Tách riêng khỏi bot chính
- Bot crypto: Gửi báo cáo BTC/ETH
- Bot từ vựng: Gửi bài học tiếng Anh
- **Hoàn toàn độc lập** nhau

## 📱 Tin nhắn mẫu
```
📚 HỌC TỪ VỰNG TIẾNG ANH HÔM NAY

1. ACHIEVE (B1)
📖 đạt được, hoàn thành
💬 "She worked hard to achieve her goals."

2. ADVENTURE (A2)
📖 cuộc phiêu lưu
💬 "The book tells the story of a great adventure."

3. ANCIENT (B1)
📖 cổ xưa, cổ đại
💬 "We visited an ancient temple in Greece."

⏰ 25/09/2025 15:09
🎯 Level: A2 - B2 (Trung cấp)
💡 Học 3 từ mỗi ngày để cải thiện tiếng Anh!
```

## 🎯 Kết quả
**Bot sẽ gửi 3 bài học từ vựng mỗi ngày (mỗi bài 5 từ) qua Telegram riêng!** 📚✨
