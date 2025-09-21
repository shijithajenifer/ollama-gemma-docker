# Local LLM Project - Dockerized

## Project Overview
This project allows you to run a **locally trained LLM (Large Language Model)** using **Docker**. You can start the model in a container without manually installing Python or dependencies. Once running, you can access it via a simple API (FastAPI + Uvicorn).  

The purpose of this project is to:
- Easily deploy a local AI model.
- Experiment with your own trained models.
- Share your project on GitHub in a ready-to-run form.

---

## Installation & Running

Step 1: Clone the repository

Open **VS Code terminal** or any command line and run:
```bash
git clone https://github.com/shijithajenifer/ollama-gemma-docker.git
cd ollama-gemma-docker

Step 2: Build and run Docker

Make sure Docker Desktop is running, then execute:
      docker-compose up --build
      
Step 3: Access the API

After the container starts, your LLM server will be running at:
     http://localhost:8000
 You can now send requests to your model via HTTP or integrate it into other applications.

## 📂 Project Structure
ollama-gemma-docker/
│── model/ # Your trained model files
│ ├── pytorch_model.bin
│ ├── config.json
│ ├── tokenizer.json
│ └── tokenizer_config.json
│── app.py # FastAPI app
│── save_model.py # Model saving script
│── requirements.txt # Python dependencies
│── Dockerfile # Docker build instructions
│── docker-compose.yml # Docker Compose setup
└── README.md # Project documentation

 Features

      - Local Deployment: Run your LLM completely on your own machine using Docker.
      - FastAPI Server: Access the model through a simple API.
      - Dockerized: No need to install Python packages manually; dependencies are handled automatically.
      - Easy Model Swap: Replace pytorch_model.bin to test different trained models.
      - Lightweight & Portable: Shareable GitHub project that works anywhere Docker is installed.
      - Real-time Testing: Send requests to the API and get immediate responses from your LLM.

  How to Use

   1. Start the server with Docker Compose.
   2. Send POST requests to the API endpoint with text input.
   3. Receive model predictions in real-time.

🛠️ Tech Stack
 
     - Python 3.10
     - PyTorch
     - FastAPI
     - Docker & Docker Compose

📌 Next Steps

     - Add support for GPU acceleration
     - Deploy on cloud (AWS / Azure / GCP)
     - Fine-tune the model on more datasets
