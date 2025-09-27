# Hướng dẫn Setup Vocabulary Bot cho GitHub Actions

## Vấn đề đã được giải quyết ✅

**Trước đây**: Vocabulary bot gửi 5 từ giống nhau mỗi lần chạy GitHub Actions
**Bây giờ**: Mỗi lần chạy sẽ gửi 5 từ khác nhau theo thứ tự tuần tự

## Cách hoạt động

1. **Local Development**: Sử dụng file `vocabulary_progress.json` để lưu progress
2. **GitHub Actions**: Tự động tải progress từ GitHub repository và lưu lại sau mỗi lần chạy

## Setup GitHub Actions

### 1. Tạo GitHub Token

1. Vào GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Tạo token mới với quyền `repo` (full control)
3. Copy token và thêm vào repository secrets

### 2. Thêm Repository Secrets

Vào repository → Settings → Secrets and variables → Actions → New repository secret:

- **Name**: `GITHUB_TOKEN`
- **Value**: [GitHub token bạn vừa tạo]

### 3. Cấu hình GitHub Actions Workflow

Tạo file `.github/workflows/vocabulary-bot.yml`:

```yaml
name: Vocabulary Bot

on:
  schedule:
    # Chạy 3 lần/ngày: 6:00, 14:00, 22:00 UTC
    - cron: '0 6,14,22 * * *'
  workflow_dispatch: # Cho phép chạy thủ công

jobs:
  vocabulary-bot:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run vocabulary bot
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        VOCABULARY_BOT_TOKEN: ${{ secrets.VOCABULARY_BOT_TOKEN }}
        VOCABULARY_CHAT_ID: ${{ secrets.VOCABULARY_CHAT_ID }}
      run: |
        python run_vocabulary_once.py
    
    - name: Commit progress (if any)
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add vocabulary_progress.json
        git diff --staged --quiet || git commit -m "Update vocabulary progress"
        git push
```

### 4. Thêm Bot Secrets

Thêm các secrets sau vào repository:

- `VOCABULARY_BOT_TOKEN`: Token của Telegram bot
- `VOCABULARY_CHAT_ID`: Chat ID để gửi tin nhắn

## Cách hoạt động của Progress System

### Local Development
```python
# File: vocabulary_progress.json
{
  "current_index": 15,
  "last_updated": "2025-09-27T17:14:19.139075",
  "total_words": 520,
  "words_sent": 5,
  "start_index": 10,
  "words_sent_this_time": ["word1", "word2", "word3", "word4", "word5"]
}
```

### GitHub Actions
1. **Lần chạy đầu tiên**: Không có progress file → bắt đầu từ index 0
2. **Lần chạy tiếp theo**: Tải progress từ GitHub → tiếp tục từ index đã lưu
3. **Sau khi gửi**: Lưu progress mới lên GitHub

## Test Vocabulary Bot

### Test Local
```bash
python vocabulary_learning.py
```

### Test GitHub Actions
```bash
python run_vocabulary_once.py
```

## Troubleshooting

### Lỗi thường gặp

1. **"ModuleNotFoundError: No module named 'schedule'"**
   - Giải pháp: Đã thêm `schedule>=1.2.0` vào requirements.txt

2. **"UnicodeEncodeError" trên Windows**
   - Giải pháp: Lỗi này chỉ ảnh hưởng console output, bot vẫn hoạt động bình thường

3. **Progress không được lưu trên GitHub**
   - Kiểm tra GITHUB_TOKEN có đúng quyền không
   - Kiểm tra repository secrets

### Debug Progress

```python
from vocabulary_learning import debug_vocabulary_progress
debug_vocabulary_progress()
```

## Kết quả

✅ **Mỗi lần chạy GitHub Actions sẽ gửi 5 từ khác nhau**
✅ **Progress được lưu trữ bền vững trên GitHub**
✅ **Không còn lỗi gửi từ trùng lặp**
✅ **Tự động reset khi học hết 520 từ**
