# Smart Food Expiry Detection & Reduction 🍎📱

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

## 🎯 Problem Statement

Food waste is a critical global issue, with households and retailers discarding significant amounts of edible food due to poor expiry tracking and lack of timely consumption reminders. Manual monitoring of expiry dates is error-prone, inefficient, and leads to both economic loss and environmental impact.

## 🚀 Objectives

- **Automated Detection**: Use computer vision to detect food items and extract expiry dates from packaging
- **Smart Prediction**: Predict remaining shelf life and freshness using time-series forecasting models
- **Proactive Notifications**: Send timely alerts before food expires to reduce waste
- **Actionable Insights**: Provide recipe suggestions and consumption recommendations for soon-to-expire items
- **Scalable Solution**: Deploy as a web application with mobile-friendly interface

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Image Input   │───▶│  Food Detection │───▶│ Expiry Prediction│
│ (Fridge Photo)  │    │   (YOLOv8)      │    │ (LSTM/Prophet)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OCR Engine    │    │  Expiry Database│    │  Notifications  │
│ (Tesseract/EasyOCR) │    │ (SQLite/JSON)   │    │(Email/WhatsApp) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Tech Stack

- **Computer Vision**: YOLOv8 (Ultralytics), OpenCV, EasyOCR
- **Machine Learning**: PyTorch, Prophet, LSTM (TensorFlow/Keras)
- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: Streamlit
- **Deployment**: Docker, Uvicorn
- **Notifications**: Twilio API, Email (SMTP)

## 📂 Project Structure

```
food-expiry-detection/
│
├── data/                    # Datasets and sample images
│   ├── sample_images/       # Sample fridge/food images
│   └── datasets.md          # Dataset sources and links
│
├── notebooks/               # Jupyter notebooks for experimentation
│   ├── food_detection_eda.ipynb
│   ├── expiry_modeling.ipynb
│   └── data_preprocessing.ipynb
│
├── src/                     # Core ML and utility code
│   ├── __init__.py
│   ├── detector.py          # YOLOv8 food detection
│   ├── data_loader.py       # Dataset loading utilities
│   ├── expiry_db.py         # Expiry database management
│   ├── forecast.py          # Freshness prediction models
│   └── utils.py             # Helper functions
│
├── api/                     # FastAPI backend
│   ├── __init__.py
│   ├── main.py              # Main FastAPI application
│   ├── models.py            # Pydantic models
│   └── routes/              # API route handlers
│
├── frontend/                # Streamlit frontend
│   ├── app.py               # Main Streamlit app
│   └── components/          # UI components
│
├── deployment/              # Docker and deployment configs
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
│
├── docs/                    # Documentation
│   ├── architecture.md
│   └── api_docs.md
│
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/food-expiry-detection.git
cd food-expiry-detection
pip install -r requirements.txt
```

### 2. Download Pre-trained Models

The system will automatically download YOLOv8 models on first run.

### 3. Run the Application

#### Option A: Streamlit Frontend
```bash
streamlit run frontend/app.py
```

#### Option B: FastAPI Backend
```bash
uvicorn api.main:app --reload
```

#### Option C: Docker
```bash
docker-compose up --build
```

## 🎯 Core Features

### 1. Food Detection
- Upload fridge images and automatically detect food items
- Extract bounding boxes and confidence scores
- Support for 100+ common food categories

### 2. Expiry Prediction
- Database of average shelf life for different food items
- Time-series forecasting for freshness degradation
- Considers storage conditions (room temperature vs. refrigerated)

### 3. Smart Notifications
- Email alerts for items nearing expiry
- WhatsApp/Telegram integration (optional)
- Customizable reminder intervals

### 4. Recipe Suggestions
- AI-powered recipe recommendations using soon-to-expire items
- Integration with recipe APIs
- Difficulty and preparation time filters

## 📊 API Endpoints

- `POST /detect` - Upload image for food detection
- `GET /expiry/{item_id}` - Get expiry prediction for specific item
- `POST /notify` - Send expiry notifications
- `GET /recipes/{items}` - Get recipe suggestions for items

## 🧪 Extensions & Future Work

- [ ] Barcode scanner integration
- [ ] Mobile app (React Native/Flutter)
- [ ] IoT sensor integration (temperature, humidity)
- [ ] Shopping list generation
- [ ] Waste analytics dashboard
- [ ] Multi-language support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Food-101 Dataset
- Open Images Dataset
- Streamlit Community

---

**Built with ❤️ for reducing food waste and creating a sustainable future.**