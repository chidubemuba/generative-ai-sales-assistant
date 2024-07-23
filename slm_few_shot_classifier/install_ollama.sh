#!/bin/bash

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Install the Ollama Python library
pip install ollama

# Pull llama3 model
ollama pull llama3

# Pull Mixtral 8x7B model
ollama pull mixtral:8x7b

# Start the Ollama server
nohup ollama serve &

# Wait for a few seconds to ensure the server starts
sleep 5

# Check if the Ollama server is running
if curl -s http://localhost:11434/v1/health; then
  echo "Ollama server is running."
else
  echo "Failed to start Ollama server."
  exit 1
fi