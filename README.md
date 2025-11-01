# 🌍 Waste Classification System

## Project Overview
This project implements an intelligent waste classification system that can identify different types of waste materials in real-time using computer vision and deep learning. The system can classify waste into six different categories: cardboard, glass, metal, paper, plastic, and trash.

## 🎯 Features
- Real-time waste classification using OpenCV
- Web interface for image upload and classification
- Detailed recycling instructions for each waste type
- High accuracy waste detection (91.2% on test set)
- Support for both image files and live camera feed
- Interactive and user-friendly interface using Streamlit

## 📊 Model Performance
The model achieves the following performance metrics:
- Overall Accuracy: 91.2%
- Training Time: ~2.5 hours (on standard GPU)
- Model Architecture: MobileNetV2 (transfer learning)
- Input Image Size: 224x224 pixels

## 🛠️ Technologies Used
- Python 3.11
- TensorFlow 2.20.0
- OpenCV (cv2)
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## 📁 Project Structure
```
project/
│
├── Data/
│   ├── Train/              # Training dataset
│   ├── Test/               # Testing dataset
│   └── README.md           # Dataset information
│
├── Deployment/
│   └── models/             # Saved model files
│
├── model.py                # Model architecture and training functions
├── training.py             # Training script
├── opencv_inference.py     # Real-time detection script
├── app.py                  # Streamlit web interface
└── requirements.txt        # Project dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11 or higher
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/waste-classification.git
cd waste-classification
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/Mac
source .venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

### Running the Application

1. For the Streamlit web interface:
```bash
streamlit run app.py
```

2. For real-time detection using camera:
```bash
python opencv_inference.py
```

3. To train the model:
```bash
python training.py
```

## 📸 Real-Time Detection
The system supports real-time waste detection using your computer's webcam:
1. Run the OpenCV inference script
2. Point your camera at waste items
3. Get instant classification results
4. Press 'q' to quit the application

## 🎓 Model Training

The model was trained on the Garbage Classification dataset from Kaggle, containing:
- cardboard (393 images)
- glass (491 images)
- metal (400 images)
- paper (584 images)
- plastic (472 images)
- trash (127 images)

Training process includes:
1. Data augmentation for better generalization
2. Transfer learning using MobileNetV2
3. Fine-tuning for optimal performance
4. Class weight balancing for uneven distribution

## 💡 Usage Tips

1. For best results:
   - Ensure good lighting conditions
   - Center the object in the frame
   - Avoid cluttered backgrounds
   - Keep the camera steady

2. Web Interface:
   - Upload clear images
   - Follow the recycling instructions
   - Check confidence scores
   - Use the provided recycling tips

## 📊 Class-wise Performance

| Category  | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Cardboard | 0.94      | 0.92   | 0.93     |
| Glass     | 0.89      | 0.91   | 0.90     |
| Metal     | 0.93      | 0.90   | 0.91     |
| Paper     | 0.92      | 0.94   | 0.93     |
| Plastic   | 0.90      | 0.88   | 0.89     |
| Trash     | 0.88      | 0.86   | 0.87     |

## 🔄 Future Improvements
- Support for more waste categories
- Mobile application development
- Multi-language support
- Integration with recycling facility databases
- Enhanced real-time performance
- Support for batch processing

## 🤝 Contributing

Contributions are welcome! Please feel free to contact the repository owner if you wish to contribute or collabroate.

## 📝 License

© 2025 Shivam Bhoyar — All rights reserved.  
This project is licensed under a **proprietary license**.  
You **may not** copy, reproduce, distribute, or use this code or its components without prior written permission.  
For licensing inquiries, contact: **shivambhoyarofficial@gmail.com**

## 📧 Contact

For any queries or suggestions, please feel free to reach out to **shivambhoyarofficial@gmail.com**.
