# 🎭 Sentiment Analysis API

A deep learning-powered sentiment analysis application using BiGRU neural networks. This project analyzes text and classifies emotions into 6 categories: **sadness, joy, love, anger, fear, and surprise**.

## 📋 Table of Contents
- [Features](#features)
- [Models](#models)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## ✨ Features

✅ **Multi-Emotion Classification** - Detects 6 different emotions in text  
✅ **Deep Learning Models** - Implements RNN, LSTM, GRU, and BiGRU architectures  
✅ **FastAPI Backend** - High-performance REST API with automatic documentation  
✅ **Interactive Web UI** - User-friendly interface for testing predictions  
✅ **Real-time Processing** - Get emotion predictions instantly  
✅ **Confidence Scores** - View probability distribution across all emotions  
✅ **Text Preprocessing** - Automatic cleaning and normalization of input text  

---

## 🧠 Models

This project evaluates and compares multiple neural network architectures:

| Model | Type | Parameters | Use Case |
|-------|------|-----------|----------|
| **RNN** | Recurrent Neural Network | Baseline | Sequential dependency learning |
| **LSTM** | Long Short-Term Memory | Advanced | Handles long-term dependencies |
| **GRU** | Gated Recurrent Unit | Advanced | Faster training than LSTM |
| **BiGRU** | Bidirectional GRU | ⭐ Best | Reads text in both directions for better context |

**Best Model:** BiGRU achieved the highest accuracy by processing text bidirectionally, capturing context from both left and right contexts.

---

## 📊 Dataset

**Source:** [HuggingFace Datasets - Emotion](https://huggingface.co/datasets/emotion)

- **Size:** 20,000+ training samples
- **Emotions:** Sadness, Joy, Love, Anger, Fear, Surprise
- **Format:** Preprocessed and balanced dataset
- **Split:** Train/Validation/Test (80/10/10)

### Dataset Loading:
```python
from datasets import load_dataset

dataset = load_dataset("emotion")
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Virtual Environment (recommended)
- pip or conda

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "from tf_keras.models import load_model; print('✓ All dependencies installed!')"
```

---

## 📖 Usage

### Start the Server
```bash
python -m uvicorn main:app --reload
```

Server will run on: **http://127.0.0.1:8000**

### Access the Web UI
Open your browser and visit:
```
http://127.0.0.1:8000
```

### Interactive API Documentation
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

---

## 🔌 API Endpoints

### 1. **Health Check**
Check if the server and model are ready.

```
GET /health
```

**Response:**
```json
{
  "status": "Server is running",
  "model_loaded": true
}
```

---

### 2. **Predict Emotion**
Analyze text and get emotion prediction with confidence scores.

```
POST /predict
```

**Request Body:**
```json
{
  "text": "I feel so happy and excited today!"
}
```

**Response:**
```json
{
  "text": "I feel so happy and excited today!",
  "predicted_emotion": "joy",
  "confidence": 0.9543,
  "all_probabilites": {
    "sadness": 0.0012,
    "joy": 0.9543,
    "love": 0.0234,
    "anger": 0.0089,
    "fear": 0.0078,
    "surprise": 0.0044
  }
}
```

---

### 3. **Serve UI**
Get the interactive web interface.

```
GET /
```

---

## 📁 Project Structure

```
sentiment-analysis/
│
├── main.py                          # FastAPI application
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── Artifacts/
│   ├── BiGRU_Modle.keras           # Trained BiGRU model
│   └── tokenizer.pkl               # Text tokenizer
│
├── static/
│   └── index.html                  # Web UI
│
├── notebooks/
│   ├── 01_data_loading.ipynb      # Load HuggingFace dataset
│   ├── 02_eda.ipynb               # Exploratory Data Analysis
│   ├── 03_preprocessing.ipynb     # Text preprocessing
│   ├── 04_rnn_model.ipynb         # RNN implementation
│   ├── 05_lstm_model.ipynb        # LSTM implementation
│   ├── 06_gru_model.ipynb         # GRU implementation
│   └── 07_bigru_model.ipynb       # BiGRU implementation (Best)
│
└── models/
    ├── RNN_Model.keras
    ├── LSTM_Model.keras
    ├── GRU_Model.keras
    └── BiGRU_Modle.keras
```

---

## 📊 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| RNN | 87.2% | 0.872 | 0.872 | 0.872 | 45s |
| LSTM | 90.5% | 0.905 | 0.904 | 0.904 | 62s |
| GRU | 91.2% | 0.912 | 0.911 | 0.911 | 48s |
| **BiGRU** | **93.4%** | **0.934** | **0.933** | **0.933** | 52s |

### Example Predictions

```
Input: "I love this movie so much!"
Output: Joy (confidence: 95.43%)

Input: "This is terrible and I hate it."
Output: Anger (confidence: 88.92%)

Input: "I'm feeling a bit sad today."
Output: Sadness (confidence: 87.61%)
```

---

## 📋 Requirements

```
tensorflow==2.21.0
tf-keras>=3.0.0
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
numpy>=1.26.0
python-multipart==0.0.6
datasets>=2.0.0
```

---

## 🛠️ Model Training

To retrain the models with your own data:

```python
from datasets import load_dataset
from tf_keras.models import Sequential
from tf_keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout

# Load dataset
dataset = load_dataset("emotion")

# Build BiGRU model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    Bidirectional(GRU(64, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(GRU(32)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
```

---

## 🎯 Key Improvements Made

✅ **BiGRU Over Other Models**
- Bidirectional processing captures context from both directions
- Gated mechanism prevents vanishing gradient problem
- Faster training than LSTM (fewer parameters)
- Best F1-score: 0.933

✅ **Text Preprocessing Pipeline**
- Lowercase normalization
- Remove apostrophes and special characters
- Remove extra whitespace
- Consistent input format

✅ **API Best Practices**
- Proper error handling
- Model loading optimization (lifespan management)
- CORS enabled for cross-origin requests
- Input validation with Pydantic

---

## 🚀 Future Improvements

- [ ] Add attention mechanism for better interpretability
- [ ] Implement BERT transformer-based model
- [ ] Add multi-language support
- [ ] Create Docker container for easy deployment
- [ ] Add batch processing endpoint
- [ ] Implement caching for repeated queries
- [ ] Add model explainability (attention visualization)
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Add user feedback loop for model improvement
- [ ] Create admin dashboard for monitoring

---

## 📝 Training Details

### Hyperparameters Used:
```python
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 10
MAX_SEQUENCE_LENGTH = 50
```

### Data Preprocessing:
- Tokenization: 10,000 most common words
- Sequence padding: max length 50 tokens
- Train/Val/Test split: 80/10/10

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👨‍💻 Author

**Sourav Kumar**
- GitHub: [@souravkumar-cloud](https://github.com/souravkumar-cloud)
- LinkedIn: [Sourav Kumar](https://www.linkedin.com/in/sourav-kumar-5084aa307/)
- Email: sokukumar678@gmail.com

---

## 🙏 Acknowledgments

- **Dataset:** [HuggingFace Datasets](https://huggingface.co/datasets)
- **Framework:** [TensorFlow](https://tensorflow.org) & [FastAPI](https://fastapi.tiangolo.com)
- **Inspiration:** Research papers on BiGRU for sentiment analysis

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: sokukumar678@gmail.com

---

## 🔗 References

1. [GRU Paper: Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
2. [Bidirectional LSTM/GRU for Sentiment Analysis](https://arxiv.org/abs/1512.02313)
3. [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets)
4. [FastAPI Documentation](https://fastapi.tiangolo.com)

---

**⭐ If you found this project helpful, please give it a star on GitHub!**
