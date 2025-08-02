#!/bin/bash

# Script to automate Python environment setup for the Image Captioning UI (trained_model_ui.py)

echo "----------------------------------------"
echo "    Trained Model UI Setup Script"
echo "----------------------------------------"

# Check for python3
if ! command -v python3 &>/dev/null; then
    echo "python3 is not installed. Please install Python 3.8 or newer."
    exit 1
fi

VENV_DIR="venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment \"$VENV_DIR\" already exists."
fi

echo ""
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
    echo "To activate your virtual environment, run:"
    echo "source $VENV_DIR/bin/activate"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    echo "To activate your virtual environment, run:"
    echo "$VENV_DIR\\Scripts\\activate"
else
    echo "Unknown OS. Please activate your virtual environment manually."
fi

echo ""
echo "Upgrading pip and installing dependencies from requirements.txt..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

echo ""
echo "Downloading the required spaCy English model..."
"$VENV_DIR/bin/python" -m spacy download en_core_web_sm

echo ""
echo "----------------------------------------"
echo "Setup complete!"
echo "Activate the virtual environment as shown above, then run:"
echo "python trained_model_ui.py"
echo "OR"
echo "streamlit run trained_model_ui.py"
echo "----------------------------------------"
