import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import os

st.set_page_config(
    page_title="Waste Classification System",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #667EEA;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .recycling-tip {
        background-color: #45A049;
        border-left: 5px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .header-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Hide sidebar */
    section[data-testid="stSidebar"] {
        display: none;
    }
    button[data-testid="baseButton-header"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

def create_complex_model():
    """Create a more complex model that matches the saved weights"""
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3)
    )
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(6, activation='softmax')
    ])
    
    return model

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def gen_labels():
    """Generate class labels"""
    return ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def load_model():
    """Load the trained model with proper architecture"""
    
    model_paths = [
        "Deployment/models/model_complete_20251102_012518.h5",
        "models/model_complete_20251102_012518.h5",
        "model_complete_20251102_012518.h5",
        "Deployment/weights/model.weights.h5",  
        "weights/model.weights.h5"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                st.success(f"✅ Model loaded successfully from {model_path}")
                return model
            except Exception as e:
                st.warning(f"⚠️ Failed to load complete model from {model_path}: {str(e)}")
                continue
    
    weights_path = "Deployment/weights/model.weights.h5"
    if os.path.exists(weights_path):
        try:
            model = create_complex_model()
            model.load_weights(weights_path)
            st.success("✅ Weights loaded with complex architecture!")
            return model
        except Exception as e:
            st.error(f"❌ Failed to load weights: {str(e)}")
    
    st.warning("🔶 Using demonstration mode (random predictions)")
    return None

def get_recycling_info(waste_type):
    """Get recycling information based on waste type"""
    recycling_info = {
        'cardboard': {
            'instructions': 'Flatten and remove any tape or labels before recycling.',
            'tips': ['Keep dry and clean', 'Break down boxes to save space'],
            'color': '#8B4513',
            'icon': '📦'
        },
        'glass': {
            'instructions': 'Rinse and remove caps before recycling. Sort by color if required.',
            'tips': ['Handle with care to avoid breakage', 'Check local guidelines for colored glass'],
            'color': "#231F23",
            'icon': '🍶'
        },
        'metal': {
            'instructions': 'Clean and remove any food residue. Separate aluminum and steel if possible.',
            'tips': ['Magnet test for steel items', 'Remove plastic components'],
            'color': "#1B1B1C",
            'icon': '🥫'
        },
        'paper': {
            'instructions': 'Remove any plastic or metal components. Keep dry and clean.',
            'tips': ['No greasy pizza boxes', 'Shredded paper may not be accepted'],
            'color': '#FFD700',
            'icon': '📄'
        },
        'plastic': {
            'instructions': 'Check the recycling number and clean before recycling. Remove caps.',
            'tips': ['Look for resin identification code', 'Flatten bottles to save space'],
            'color': '#FF69B4',
            'icon': '🧴'
        },
        'trash': {
            'instructions': 'This item should be disposed of in regular waste. Consider reduction alternatives.',
            'tips': ['Reduce consumption when possible', 'Look for reusable alternatives'],
            'color': '#696969',
            'icon': '🗑️'
        }
    }
    
    return recycling_info.get(waste_type, {
        'instructions': 'No specific instructions available.',
        'tips': [],
        'color': '#000000',
        'icon': '❓'
    })

if 'model' not in st.session_state:
    with st.spinner('Loading waste classification model...'):
        st.session_state.model = load_model()

labels = gen_labels()


st.markdown("""
    <div class="header-section">
        <h1>♻️ Waste Classification System</h1>
        <p style="font-size: 1.3rem; opacity: 0.9;">Upload an image to classify waste materials and get recycling instructions</p>
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    file = st.file_uploader('Choose an image file', 
                           type=['jpg', 'png', 'jpeg', 'bmp'],
                           help="Supported formats: JPG, PNG, JPEG, BMP")
    
    if file is not None:
        image = Image.open(file)
        st.image(image, caption='Uploaded Image', use_container_width=True)         
        if st.button('🔍 Classify Waste', type='primary'):
            with st.spinner('Analyzing image...'):

                processed_image = preprocess_image(image)

                if st.session_state.model is not None:
                    try:
                        prediction = st.session_state.model.predict(processed_image, verbose=0)
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        prediction = np.random.dirichlet(np.ones(6), size=1)
                else:
                    prediction = np.random.dirichlet(np.ones(6), size=1)
                
                predicted_class = labels[np.argmax(prediction[0])]
                
                st.session_state.prediction = prediction
                st.session_state.predicted_class = predicted_class

with col2:
    if 'prediction' in st.session_state:
        st.subheader("🎯 Classification Result")
        
        info = get_recycling_info(st.session_state.predicted_class)
        
        st.markdown(f"""
            <div class="result-box">
                <h3>{info['icon']} Classification Result</h3>
                <p style="font-size: 1.8rem; color: {info['color']}; font-weight: bold; margin: 1rem 0;">
                    {st.session_state.predicted_class.upper()}
                </p>
                <p style="font-size: 1.1rem; color: #666;">Waste Category</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("♻️ Recycling Instructions")
        st.markdown(f"""
            <div class="recycling-tip">
                <h4>📝 What to do:</h4>
                <p style="font-size: 1.1rem;">{info['instructions']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if info['tips']:
            st.markdown("**💡 Helpful Tips:**")
            for tip in info['tips']:
                st.markdown(f"- {tip}")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "♻️ Waste Classification System | Help make our planet greener 🌍"
    "</div>",
    unsafe_allow_html=True
)

if st.session_state.model is None:
    st.warning(
        "⚠️ **Demonstration Mode**: Using sample predictions. "
        "For full functionality, ensure the model file is available."
    )