# Image Captioning with Emotional Styles

A deep learning project for generating emotionally styled captions for images using a CNN+LSTM architecture. The model is trained on real MS COCO captions (neutral) and VIST-style emotional stories (happy, sad, melancholic, ecstatic, devastated). The project includes a Streamlit web app for interactive caption generation.

---

## 📂 Project Structure

```
.
├── app.py                # Streamlit web app for caption generation
├── ml.ipynb              # Jupyter notebook for data processing, training, and analysis
├── models/               # Trained model files (.h5)
├── cache/                # Tokenizer and artifacts (.pkl)
├── data/                 # COCO dataset and sample images
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🚀 Features

- **Image Captioning**: Generates captions for images using deep learning.
- **Emotional Styles**: Supports 6 styles: happy, neutral, sad, ecstatic, melancholic, devastated.
- **Real Data**: Trained on 1000+ real MS COCO captions and 500+ VIST-style emotional stories.
- **Interactive Web App**: Upload an image and select an emotion to generate a styled caption.
- **Comprehensive Analysis**: Training graphs and dataset analysis included in the notebook.

---

## 🛠️ Setup

1. **Clone the repository**
    ```bash
    git clone <repo-url>
    cd image-processing-task-3
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare Data**
    - The notebook (`ml.ipynb`) will automatically download and process the MS COCO dataset and generate VIST-style stories.

4. **Train the Model**
    - Run all cells in `ml.ipynb` to process data, train the model, and save artifacts.

5. **Run the Web App**
    ```bash
    streamlit run app.py
    ```
    - Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Model Overview

- **Encoder**: InceptionV3 CNN (pre-trained, frozen)
- **Decoder**: LSTM with style embedding
- **Tokenization**: Keras Tokenizer
- **Emotional Style**: One-hot embedding for 6 styles

---

## 📊 Training & Analysis

- **Notebook (`ml.ipynb`)** includes:
    - Data loading and cleaning
    - Feature extraction
    - Model training with early stopping
    - Training/validation loss & accuracy graphs
    - Dataset and style distribution analysis

---

## 📦 Artifacts

- `models/style_caption_model.h5` - Main captioning model
- `models/image_encoder.h5` - CNN encoder
- `models/context_encoder.h5` - Style context encoder
- `cache/caption_tokenizer.pkl` - Tokenizer for text
- `cache/caption_artifacts.pkl` - Style mappings and config

---

## 🌐 Web App Usage

1. Upload an image (JPG/PNG).
2. Select an emotional style.
3. Click "Generate Caption".
4. View and copy the generated caption.

---

## 📑 Notes

- All training data is real (no dummy data).
- Minimum of 1000 MSCOCO and 500 VIST samples are enforced.
- The project is ready for deployment and further research.

---

## 🤝 Acknowledgements

- [MS COCO](https://cocodataset.org/) for image captions
- [VIST](https://visionandlanguage.net/VIST/) for story inspiration
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for deep learning
