#!/bin/bash

# Deployment script for RAG Chatbot

echo "🚀 Starting RAG Chatbot Deployment..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Current: $python_version"
    exit 1
fi

echo "✅ Python version check passed"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p vector_db

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo "Please create .env file with your OPENAI_API_KEY"
    exit 1
fi

# Run document ingestion
echo "📄 Processing documents..."
python -c "from ingest import create_sample_documents; create_sample_documents()"

# Start Streamlit app
echo "🎯 Starting Streamlit app..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0