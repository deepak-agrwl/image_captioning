# image_captioning
Multimodal Media Retrieval and Captioning System

Project Details: https://drive.google.com/drive/folders/1E1tF9fLXq_i4XiHLJrGtp-B4UiqtWioU?usp=drive_link 

# Steps project campstone
## Installation
1. Install VS code
2. Required extensions in VS Code
    1. Python (by Microsoft)

## Run in VS code terminal
> code --install-extension ms-python.python
Note: Error: “zsh: command not found: code”
Resolution: Open the command palette in VS Code(⇧⌘P or Ctrl+Shift+P). Type and select: “Shell Command: Install 'code' command in PATH”. This adds the code command to your terminal path.

    2. Jupyter (by Microsoft)
> code --install-extension ms-toolsai.jupyter

3. Optional but Helpful Extensions (these may already part of Jupyter(by Microsoft))
    - Pylance (Fast, intelligent language support for Python, Improves IntelliSense/autocompletion dramatically)
> code --install-extension ms-python.vscode-pylance --force
    - Jupyter Keymap (Brings Jupyter-style keyboard shortcuts to VS Code notebooks)
> code --install-extension ms-toolsai.jupyter-keymap --force
    - Jupyter Notebook Renderers (Improves output rendering (LaTeX, Plotly, Vega, etc.))
> code --install-extension ms-toolsai.jupyter-renderers --force

Test It
1. Open or create a .ipynb file in VS Code.
2. Select a Python interpreter when prompted (click top-right kernel selection).
3. Run cells using Shift + Enter.
￼


# Download Kaggle Dataset Locally

1. Install the Kaggle Python Package
> pip install kaggle
Successfully installed kaggle-1.7.4.5
2. Get Your Kaggle API Token
    * Go to your Kaggle Account Settings.
    * Scroll to the API section.
    * Click "Create New API Token".
    * A file called kaggle.json will download (contains your API credentials).
3. Place kaggle.json in the Right Location
    * Place in Default Location
> mkdir ~/.kaggle
> mv ~/Downloads/kaggle.json  ~/.kaggle/
> chmod 600 ~/.kaggle/kaggle.json
4. Download a Dataset Using Python (refer ~/Documents/Project Campstone/Captioning_Campstone_Group2.ipynb)

