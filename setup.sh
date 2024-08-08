# setup.sh
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml

# Install system dependencies
apt-get update && apt-get install -y python3-distutils

# Install other necessary packages
apt-get install -y libgl1-mesa-glx

# Update pip
pip install --upgrade pip
