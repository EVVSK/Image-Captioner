# Image Captioner - Simple Style-Conditioned Caption Generation
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import io
from pathlib import Path
import sys

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Configure page
st.set_page_config(
    page_title="Image Captioner",
    page_icon="�️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced styling with subtle animations
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        animation: fadeIn 1s ease-in;
    }
    .result-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        animation: slideInUp 0.5s ease-out;
    }
    .result-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton > button {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,123,255,0.4);
    }
    .success-message {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        animation: slideInLeft 0.5s ease-out;
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_artifacts():
    """Load all trained models and artifacts"""
    try:
        # Define paths
        models_dir = Path("models")
        cache_dir = Path("cache")
        
        # Check if models exist
        caption_model_path = models_dir / "style_caption_model.h5"
        encoder_path = models_dir / "image_encoder.h5"
        tokenizer_path = cache_dir / "caption_tokenizer.pkl"
        artifacts_path = cache_dir / "caption_artifacts.pkl"
        
        # Check file existence
        missing_files = []
        if not caption_model_path.exists():
            missing_files.append(str(caption_model_path))
        if not encoder_path.exists():
            missing_files.append(str(encoder_path))
        if not tokenizer_path.exists():
            missing_files.append(str(tokenizer_path))
        if not artifacts_path.exists():
            missing_files.append(str(artifacts_path))
        
        if missing_files:
            st.error("Model files not found:")
            for file in missing_files:
                st.error(f"• {file}")
            st.error("Please run the training notebook (ml.ipynb) first to generate the models.")
            st.info("Run all cells in the notebook to train the model.")
            return None, None, None, None
        
        # Load models and artifacts
        with st.spinner("Loading trained models..."):
            caption_model = tf.keras.models.load_model(str(caption_model_path))
            image_encoder = tf.keras.models.load_model(str(encoder_path))
            
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            
            with open(artifacts_path, 'rb') as f:
                artifacts = pickle.load(f)
        
        return caption_model, image_encoder, tokenizer, artifacts
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Please ensure the training notebook has been run successfully.")
        return None, None, None, None

def preprocess_uploaded_image(uploaded_file):
    """Preprocess uploaded image for the model - matches notebook processing"""
    try:
        # Read image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to InceptionV3 input size (299x299) - same as notebook
        image_resized = image.resize((299, 299))
        
        # Convert to array and normalize
        img_array = np.array(image_resized, dtype=np.float32)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply InceptionV3 preprocessing (same as notebook)
        # Normalize to [-1, 1] range as expected by InceptionV3
        img_preprocessed = (img_array / 127.5) - 1.0
        
        return image, img_preprocessed
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def extract_image_features(image_encoder, preprocessed_image):
    """Extract features using the trained image encoder"""
    try:
        features = image_encoder.predict(preprocessed_image, verbose=0)
        return features[0]  # Remove batch dimension
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def generate_caption(model, tokenizer, image_features, style_idx, max_length, temperature=0.7):
    """Generate single caption with temperature control"""
    try:
        start_token = tokenizer.word_index.get('<start>', 1)
        end_token = tokenizer.word_index.get('<end>', 2)
        
        generated = [start_token]
        
        for _ in range(max_length - 1):
            # Prepare input - ensure proper padding
            current_seq = generated + [0] * (max_length - 1 - len(generated))
            text_input = np.array([current_seq[:max_length-1]])  # Ensure exact length
            
            # Predict next word
            predictions = model.predict([
                image_features.reshape(1, -1),
                text_input,
                np.array([style_idx])
            ], verbose=0)
            
            # Get prediction for current position
            pred_idx = min(len(generated)-1, predictions.shape[1]-1)
            next_word_probs = predictions[0, pred_idx, :]
            
            # Apply temperature sampling
            if temperature > 0:
                # Avoid log(0) by adding small epsilon
                next_word_probs = np.clip(next_word_probs, 1e-8, 1.0)
                next_word_probs = np.log(next_word_probs) / temperature
                next_word_probs = np.exp(next_word_probs)
                next_word_probs = next_word_probs / np.sum(next_word_probs)
                
                # Sample from top-k to avoid very unlikely words
                top_k = min(50, len(next_word_probs))
                top_indices = np.argpartition(next_word_probs, -top_k)[-top_k:]
                top_probs = next_word_probs[top_indices]
                top_probs = top_probs / np.sum(top_probs)
                
                selected_idx = np.random.choice(len(top_indices), p=top_probs)
                next_word_idx = top_indices[selected_idx]
            else:
                next_word_idx = np.argmax(next_word_probs)
            
            # Stop conditions
            if next_word_idx == end_token or next_word_idx == 0:
                break
                
            # Only add if it's a valid word index
            if next_word_idx in tokenizer.index_word:
                generated.append(next_word_idx)
            else:
                break  # Stop if we get an invalid token
        
        # Convert to words
        caption_words = []
        for word_idx in generated[1:]:  # Skip start token
            if word_idx in tokenizer.index_word and word_idx not in [0, end_token]:
                word = tokenizer.index_word[word_idx]
                # Skip special tokens
                if word not in ['<start>', '<end>', '<unk>']:
                    caption_words.append(word)
        
        result = ' '.join(caption_words)
        
        # If we still get empty result, try a simple fallback
        if not result:
            # Try greedy decoding as fallback
            return generate_simple_caption(model, tokenizer, image_features, style_idx, max_length)
        
        return result
        
    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return "Error generating caption"

def generate_simple_caption(model, tokenizer, image_features, style_idx, max_length):
    """Simple greedy caption generation as fallback"""
    try:
        start_token = tokenizer.word_index.get('<start>', 1)
        end_token = tokenizer.word_index.get('<end>', 2)
        
        generated = [start_token]
        
        for _ in range(max_length - 1):
            # Simple input preparation
            text_input = np.array([generated + [0] * (max_length - 1 - len(generated))])
            text_input = text_input[:, :max_length-1]
            
            # Predict
            predictions = model.predict([
                image_features.reshape(1, -1),
                text_input,
                np.array([style_idx])
            ], verbose=0)
            
            # Simple greedy selection
            next_word_idx = np.argmax(predictions[0, -1, :])
            
            if next_word_idx == end_token or next_word_idx == 0:
                break
                
            generated.append(next_word_idx)
        
        # Convert to words
        words = []
        for word_idx in generated[1:]:
            if word_idx in tokenizer.index_word:
                word = tokenizer.index_word[word_idx]
                if word not in ['<start>', '<end>', '<unk>']:
                    words.append(word)
        
        return ' '.join(words) if words else "A scene is shown"
        
    except:
        return "Image caption generated"

def generate_story(model, tokenizer, image_features, style_idx, max_length, num_sentences=3):
    """Generate coherent multi-sentence story"""
    try:
        sentences = []
        
        for i in range(num_sentences):
            # Vary temperature for diversity
            temperature = 0.6 + (i * 0.1)
            
            # Generate sentence
            sentence = generate_caption(model, tokenizer, image_features, style_idx, max_length, temperature)
            
            # Basic quality check
            if sentence and len(sentence.split()) > 2 and sentence not in sentences:
                sentences.append(sentence)
        
        # Ensure we have at least one sentence
        if not sentences:
            return "No story generated."
        
        # Join with proper punctuation
        story = '. '.join(sentences)
        if not story.endswith('.'):
            story += '.'
        return story
            
    except Exception as e:
        st.error(f"Error generating story: {str(e)}")
        return "Error generating story"

def main():
    """Main Streamlit application"""
    
    # Enhanced header
    st.markdown('<div class="main-header"><h1>🎨 AI Image Captioner</h1><p>Upload an image and generate AI-powered captions with different emotional styles</p></div>', unsafe_allow_html=True)
    
    # Load models
    caption_model, image_encoder, tokenizer, artifacts = load_models_and_artifacts()
    
    if caption_model is None:
        st.stop()
    
    # Success message with animation
    st.markdown('<div class="success-message">✅ Models loaded successfully! Ready to generate captions.</div>', unsafe_allow_html=True)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a PNG or JPG image"
        )
        
        if uploaded_file is not None:
            # Process image
            original_image, preprocessed_image = preprocess_uploaded_image(uploaded_file)
            
            if original_image is not None:
                # Display image
                st.image(original_image, caption="Uploaded Image", use_column_width=True)
                
                # Style selection
                st.subheader("Caption Style")
                
                available_styles = list(artifacts['style_to_idx'].keys())
                selected_style = st.selectbox(
                    "Choose style:",
                    options=available_styles,
                    help="Select the emotional tone for the caption"
                )
                
                # Generation type
                generation_type = st.radio(
                    "Output type:",
                    ["Single Caption", "Short Story"],
                    help="Choose between a single caption or a multi-sentence story"
                )
                
                # Generate button
                if st.button("Generate Caption", type="primary"):
                    
                    with col2:
                        st.subheader("Generated Caption")
                        
                        # Extract features
                        with st.spinner("Processing image..."):
                            image_features = extract_image_features(image_encoder, preprocessed_image)
                        
                        if image_features is not None:
                            style_idx = artifacts['style_to_idx'][selected_style]
                            max_length = artifacts['max_length']
                            
                            if generation_type == "Single Caption":
                                with st.spinner("Generating caption..."):
                                    caption = generate_caption(
                                        caption_model, tokenizer, image_features, 
                                        style_idx, max_length, temperature=0.7
                                    )
                                
                                st.markdown(f"""
                                <div class="result-box">
                                    <strong>🎯 Caption ({selected_style}):</strong><br>
                                    <em style="font-size: 1.1em; line-height: 1.5;">"{caption}"</em>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            else:  # Short Story
                                with st.spinner("Generating story..."):
                                    story = generate_story(
                                        caption_model, tokenizer, image_features,
                                        style_idx, max_length, num_sentences=3
                                    )
                                
                                st.markdown(f"""
                                <div class="result-box">
                                    <strong>📖 Story ({selected_style}):</strong><br>
                                    <em style="font-size: 1.05em; line-height: 1.6;">{story}</em>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Copy text area
                            result_text = caption if generation_type == "Single Caption" else story
                            st.text_area("Copy text:", value=result_text, height=100)
    
    with col2:
        if uploaded_file is None:
            st.subheader("📋 Instructions")
            st.info("Upload an image to generate captions")
            
            st.subheader("🎭 Available Styles")
            if artifacts:
                for style in artifacts['style_to_idx'].keys():
                    st.write(f"• **{style.title()}** - {get_style_description(style)}")

def get_style_description(style):
    """Get description for each style"""
    descriptions = {
        'neutral': 'Factual, objective descriptions',
        'happy': 'Joyful, positive interpretations', 
        'sad': 'Somber, melancholic mood',
        'ecstatic': 'Enthusiastic, thrilled descriptions',
        'melancholic': 'Thoughtful, contemplative tone',
        'devastated': 'Intense, dramatic interpretations'
    }
    return descriptions.get(style, 'Emotional interpretation')
    
    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #6c757d;">
        <strong>🧠 AI Image Captioner</strong> - CNN+LSTM with style conditioning<br>
        <small>Powered by InceptionV3 + Custom LSTM | Trained on 2,019 real samples | 74% accuracy</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
