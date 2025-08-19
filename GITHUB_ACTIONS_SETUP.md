# Hướng dẫn thiết lập GitHub Actions cho Trading Bot

## 🚀 Tự động hóa với GitHub Actions

GitHub Actions sẽ tự động chạy bot trading của bạn và gửi kết quả về Telegram mà không cần chạy script trên local.

## 📋 Các bước thiết lập

### 1. **Thiết lập GitHub Secrets**

Vào repository của bạn trên GitHub:
1. Vào **Settings** → **Secrets and variables** → **Actions**
2. Thêm 2 secrets sau:

```
TELEGRAM_BOT_TOKEN = 7496162935:AAGncIsO4q18cOWRGpK0vYb_5zWxYNEgWKQ
TELEGRAM_CHAT_ID = 1866335373
```

### 2. **Cấu hình Workflow**

File `.github/workflows/run-script.yml` đã được tạo sẵn với các tính năng:

- ✅ **Tự động chạy mỗi 2 giờ** (cron: '0 */2 * * *')
- ✅ **Chạy khi push code mới** 
- ✅ **Chạy thủ công** (workflow_dispatch)
- ✅ **Cài đặt dependencies tự động**
- ✅ **Gửi kết quả về Telegram**
- ✅ **Upload logs khi có lỗi**

### 3. **Lịch chạy tự động**

Bot sẽ chạy tự động:
- **Mỗi 2 giờ** (00:00, 02:00, 04:00, 06:00, 08:00, 10:00, 12:00, 14:00, 16:00, 18:00, 20:00, 22:00 UTC)
- **Khi bạn push code mới** lên branch main/master
- **Khi bạn trigger thủ công** từ GitHub Actions tab

### 4. **Kiểm tra hoạt động**

1. Vào **Actions** tab trên GitHub repository
2. Bạn sẽ thấy workflow "Trading Bot Analysis"
3. Click vào workflow để xem logs và kết quả

## 🔧 Tính năng của Workflow

### **Tự động cài đặt:**
- Python 3.11
- Tất cả dependencies từ requirements.txt
- System dependencies cần thiết
- Tạo các thư mục cần thiết (ml_models, ml_data, prediction_data)

### **Xử lý lỗi:**
- Timeout sau 30 phút
- Upload logs và data khi có lỗi
- Upload kết quả khi thành công

### **Bảo mật:**
- Sử dụng GitHub Secrets cho Telegram credentials
- Không expose sensitive data trong logs

## 📱 Kết quả nhận được

Bot sẽ gửi về Telegram:
- ✅ **Phân tích Technical Analysis** cho BTC/USDT và ETH/USDT
- ✅ **Smart Money Concepts (SMC)** patterns
- ✅ **Divergence Analysis** 
- ✅ **Convergence Analysis**
- ✅ **Machine Learning predictions** (nếu models đã train)
- ✅ **Multi-timeframe consensus**
- ✅ **Entry points, Stop Loss, Take Profit**
- ✅ **Risk/Reward ratios**

## 🛠️ Troubleshooting

### **Nếu workflow fail:**

1. **Kiểm tra logs** trong Actions tab
2. **Kiểm tra Telegram credentials** trong Secrets
3. **Kiểm tra dependencies** trong requirements.txt
4. **Kiểm tra code** có lỗi syntax không

### **Nếu không nhận được Telegram message:**

1. **Kiểm tra bot token** có đúng không
2. **Kiểm tra chat ID** có đúng không  
3. **Kiểm tra bot có quyền** gửi message không
4. **Kiểm tra internet connection** của GitHub runner

## 📊 Monitoring

### **Theo dõi performance:**
- Vào Actions tab để xem execution time
- Kiểm tra logs để debug issues
- Monitor Telegram messages để đảm bảo bot hoạt động

### **Cập nhật code:**
- Push code mới lên main/master branch
- Workflow sẽ tự động chạy với code mới
- Không cần restart hay cấu hình lại

## 🎯 Lợi ích

✅ **Không cần máy tính local chạy 24/7**
✅ **Tự động cập nhật khi có code mới**
✅ **Scalable và reliable**
✅ **Free với GitHub public repositories**
✅ **Dễ dàng monitor và debug**
✅ **Bảo mật với GitHub Secrets**

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra logs trong Actions tab
2. Đảm bảo Telegram credentials đúng
3. Kiểm tra code không có lỗi syntax
4. Đảm bảo tất cả dependencies được cài đặt

---

**🎉 Chúc mừng! Bot của bạn giờ đây sẽ chạy tự động và gửi kết quả về Telegram mà không cần can thiệp thủ công!**
