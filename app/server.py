from fastapi import FastAPI
import torch
import torch.nn as nn

app = FastAPI()

# -----------------------
# Load dummy model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    def forward(self, x):
        return self.linear(x)

# model path inside container
model = DummyModel()
model.load_state_dict(torch.load("model/pytorch_model.bin"))
model.eval()

@app.get("/")
def home():
    return {"message": "LLM server is running!"}

@app.post("/generate")
def generate(input: dict):
    # dummy input/output
    x = torch.randn(1, 10)
    output = model(x)
    return {"output": output.tolist()}
