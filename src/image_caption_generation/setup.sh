pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install --upgrade certifi
mkdir -p ../../resources/models
if [ ! -f ../../resources/models/resnet50-19c8e357.pth ]; then
  curl -k -o ../../resources/models/resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth
fi