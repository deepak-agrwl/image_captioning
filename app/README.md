# Trained Model UI — Image Captioning Application

This user interface allows you to generate captions for your images using a pre-trained Vision Transformer + GPT2 image captioning model.

## Quick Start

### 1. Move to the App Directory

```sh
cd app
```

### 3. Install Dependencies

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

Or simply run:

```sh
bash setup.sh
```

### 4. Download spaCy English Model (First Time Only)

The application requires the spaCy "en_core_web_sm" model. Run:

```sh
python -m spacy download en_core_web_sm
```

### 5. Run the UI

```sh
python trained_model_ui.py
```

or if using Streamlit:

```sh
streamlit run trained_model_ui.py
```

---

## Notes

- Requires Python 3.8 or newer.
- GPU-accelerated hardware is strongly recommended for best performance.
- Make sure the trained model file (`models/gpt2_vit.pth`) is present in the `app/models/` folder before running.
