# Test SSL in your virtual environment
import ssl
import urllib.request
import certifi

# Test if certificates work
try:
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen('https://www.google.com', context=context) as response:
        print("SSL certificates working!")
except Exception as e:
    print(f"SSL Error: {e}")