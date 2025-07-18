# 🤟 Real-Time Hand Sign Detection with PyTorch

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-brightgreen.svg)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange.svg)](https://opencv.org)

> **A sophisticated real-time American Sign Language (ASL) recognition system powered by deep learning and computer vision technologies.**

## 🎯 Project Overview

This project implements an intelligent hand sign detection system that recognizes ASL letters (A, B, C) in real-time using a custom-trained neural network. The system offers both a modern web interface and a traditional computer vision application, making it accessible for various use cases.

### ✨ Key Features

- **🔥 Real-Time Detection**: Instant hand sign recognition with live camera feed
- **🌐 Web Interface**: Beautiful Streamlit-powered web application
- **🖥️ Desktop Application**: OpenCV-based standalone application
- **🧠 Custom Neural Network**: Purpose-built PyTorch model for ASL recognition
- **📝 Interactive Typing**: Convert hand signs to text with confirmation system
- **🎨 Modern UI/UX**: Intuitive interface with real-time feedback

## 🏗️ Architecture

### Deep Learning Model
- **Framework**: PyTorch
- **Architecture**: Custom 3-layer fully connected neural network
- **Input**: 300x300 grayscale images (90,000 features)
- **Output**: 3 classes (A, B, C)
- **Preprocessing**: Automatic hand detection, cropping, and normalization

### Computer Vision Pipeline
1. **Hand Detection**: MediaPipe/CVZone hand tracking
2. **Image Preprocessing**: Grayscale conversion, resizing, normalization
3. **Feature Extraction**: Flattened pixel values
4. **Classification**: Neural network inference
5. **Post-processing**: Confidence-based filtering and display

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Webcam/Camera access
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Real-Time-Hand-Sign-Detection-Pytorch.git
   cd Real-Time-Hand-Sign-Detection-Pytorch
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the applications**

   **Web Application (Recommended)**
   ```bash
   streamlit run app.py
   ```
   
   **Desktop Application**
   ```bash
   python main.py
   ```

## 💻 Usage

### Web Application
1. Launch the Streamlit app
2. Allow camera permissions
3. Show hand signs (A, B, or C) to the camera
4. View real-time predictions
5. Use "Confirm Letter" to build text
6. Use "Delete Last Letter" to correct mistakes

### Desktop Application
1. Run the Python script
2. Position your hand in front of the camera
3. View predictions overlaid on the video feed
4. Press 'q' to quit

## 📁 Project Structure

```
Real-Time-Hand-Sign-Detection-Pytorch/
├── 📱 app.py                 # Streamlit web application
├── 🖥️ main.py               # OpenCV desktop application
├── 🧠 model.py               # Neural network architecture & utilities
├── 🔧 hand_tracker.py        # Hand detection utilities
├── ⚙️ preprocess.py          # Image preprocessing functions
├── 📊 nn_model.ipynb         # Model training notebook
├── 🎯 hand_sign_model.pth    # Trained model weights
├── 📈 data.csv               # Training dataset
├── 📋 requirements.txt       # Project dependencies
└── 📖 README.md              # Project documentation
```

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Deep Learning** | PyTorch | Neural network framework |
| **Computer Vision** | OpenCV, MediaPipe | Image processing & hand detection |
| **Web Framework** | Streamlit | Interactive web interface |
| **Hand Tracking** | CVZone | Advanced hand detection |
| **Data Processing** | NumPy | Numerical computations |
| **Real-time Streaming** | streamlit-webrtc | Web camera integration |

## 🎯 Model Performance

- **Architecture**: 3-layer fully connected network
- **Input Size**: 90,000 features (300x300 grayscale)
- **Classes**: 3 (A, B, C hand signs)
- **Inference Speed**: Real-time (30+ FPS)
- **Model Size**: ~185MB

## 🔧 Configuration

### Model Parameters
```python
# Neural Network Architecture
Input Layer: 90,000 neurons
Hidden Layer 1: 512 neurons (ReLU + Dropout 0.3)
Hidden Layer 2: 256 neurons (ReLU + Dropout 0.3)
Output Layer: 3 neurons (Softmax)
```

### Image Processing
```python
# Preprocessing Pipeline
Image Size: 300x300 pixels
Color Space: Grayscale
Normalization: [0, 1] range
Hand Detection: MediaPipe/CVZone
```

## 🚀 Future Enhancements

- [ ] **Extended Alphabet**: Support for full A-Z ASL alphabet
- [ ] **Word Recognition**: Detect complete words and phrases
- [ ] **Mobile App**: React Native/Flutter mobile application
- [ ] **Cloud Deployment**: AWS/GCP hosted web service
- [ ] **Model Optimization**: TensorRT/ONNX optimization for edge devices
- [ ] **Multi-language Support**: International sign language variants
- [ ] **Voice Synthesis**: Text-to-speech integration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Developer

**Harsh** - *Machine Learning Engineer*

- 🔗 [LinkedIn](https://linkedin.com/in/yourprofile)
- 📧 [Email](mailto:your.email@example.com)
- 🐙 [GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- **MediaPipe Team** for robust hand detection algorithms
- **PyTorch Community** for the excellent deep learning framework
- **Streamlit Team** for the intuitive web app framework
- **OpenCV Contributors** for computer vision utilities

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

*Built with ❤️ for the deaf and hard-of-hearing community*

</div>
