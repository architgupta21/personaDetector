# 🚀 Cloud Deployment Guide

This guide outlines two ways to host your Streamlit RAG Chatbot in the cloud so that the evaluation team can test it online.

---

## 🛠️ Option A: Hugging Face Spaces (Recommended & 100% Free)

Hugging Face Spaces is the easiest and most cost-effective way to host a local LLM application for free. Using a custom **Docker Space**, we can run both the Ollama server and the Streamlit frontend in a single container.

### Step 1: Create a Hugging Face Space
1. Log in to [Hugging Face](https://huggingface.co/).
2. Click on **New Space**.
3. Choose a name (e.g., `persona-rag-chatbot`).
4. Select **Docker** as the SDK (instead of Streamlit).
5. Choose the **Blank** template.
6. Set the space to **Public** so evaluators can access it.
7. Click **Create Space**.

### Step 2: Add Configuration Files
You need to add three files to your Space's repository: `Dockerfile`, `entrypoint.sh`, and your existing python files.

#### 📄 `Dockerfile`
Create a `Dockerfile` in the root of your repository:
```dockerfile
FROM python:3.10-slim

# Install system dependencies and curl
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set environment variables for offline model caching
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (including pre-populated database and persona)
COPY . .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Add entrypoint script
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
```

#### 📄 `entrypoint.sh`
Create an execution script to boot the Ollama service and Streamlit:
```bash
#!/bin/bash

# Start Ollama in the background
echo "Starting Ollama server..."
ollama serve > ollama.log 2>&1 &

# Wait for Ollama to wake up
echo "Waiting for Ollama to initialize..."
until curl -s http://127.0.0.1:11434/api/tags > /dev/null; do
    sleep 1
done

# Pull the Phi-3.5 model
echo "Downloading Phi-3.5 model..."
ollama pull phi3.5

echo "Ollama is ready! Launching Streamlit chatbot..."
# Streamlit runs on HF Space's default port 7860
python -m streamlit run chatbot.py --server.port 7860 --server.address 0.0.0.0
```

### Step 3: Push Your Code
Commit and push your files (including `conversations.csv`, `persona.json`, and the `chroma_db/` folder) to the Hugging Face Space repository.
* HF Spaces will automatically detect the `Dockerfile`, build the container, pull the model, and serve the application at your Space's URL:
  `https://huggingface.co/spaces/<your-username>/<your-space-name>`

---

## ☁️ Option B: Cloud VPS Deployment (AWS, GCP, or DigitalOcean)

If you prefer to deploy on a dedicated Virtual Private Server (VPS) (e.g., an AWS EC2 instance, GCP VM, or DigitalOcean Droplet with at least 8GB RAM), follow these steps:

### Step 1: Install Ollama & Pull Model
SSH into your server and run:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
sudo systemctl start ollama

# Pull Phi-3.5
ollama pull phi3.5
```

### Step 2: Set Up the Codebase
Clone your repo and install dependencies:
```bash
git clone <your-repo-url>
cd personaExtractor

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Configure Streamlit Service
Create a systemd service file to keep Streamlit running persistently in the background:
```bash
sudo nano /etc/systemd/system/streamlit.service
```
Paste the following:
```ini
[Unit]
Description=Streamlit Chatbot
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/personaExtractor
ExecStart=/home/ubuntu/personaExtractor/.venv/bin/streamlit run chatbot.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
Restart=always

[Install]
WantedBy=multi-user.target
```
Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl start streamlit
```

### Step 4: Configure Nginx Reverse Proxy (SSL)
To serve the app securely over `https://` on standard port `80` / `443`:
1. Install Nginx:
   ```bash
   sudo apt install nginx -y
   ```
2. Configure proxy settings:
   ```bash
   sudo nano /etc/nginx/sites-available/default
   ```
   Replace the server block with:
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com; # Replace with your domain or IP

       location / {
           proxy_pass http://127.0.0.1:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```
3. Restart Nginx:
   ```bash
   sudo systemctl restart nginx
   ```
4. Obtain an SSL Certificate with Certbot:
   ```bash
   sudo apt install certbot python3-certbot-nginx -y
   sudo certbot --nginx -d yourdomain.com
   ```
Now your app is live and secure at `https://yourdomain.com`!
