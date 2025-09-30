# Cấu hình bot Telegram riêng cho từ vựng
VOCABULARY_BOT_TOKEN = "8085678497:AAFPzATgWObijU3o1TJyUY9ukfPPGR06Tps"
VOCABULARY_CHAT_ID = "1866335373"

# Cấu hình lịch gửi từ vựng
VOCABULARY_SCHEDULE = {
    'enabled': True,
    'interval_hours': 12,  # Gửi mỗi 12 giờ
    'times': [8, 20],  # Giờ gửi (UTC) - 2 lần/ngày
    'words_per_lesson': 5  # 5 từ mỗi bài
}

# Thông tin bot
BOT_INFO = {
    'name': 'English Vocabulary Bot',
    'description': 'Bot học từ vựng tiếng Anh B1-C1',
    'version': '1.0',
    'author': 'BOT-idicator Team'
}
