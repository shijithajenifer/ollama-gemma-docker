# save_model_dummy.py
import os
import torch
import json
import torch.nn as nn

# -----------------------
# STEP 1: Create a tiny dummy model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    def forward(self, x):
        return self.linear(x)

model = DummyModel()

# -----------------------
# STEP 2: Save weights
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/pytorch_model.bin")
print("Saved pytorch_model.bin")

# -----------------------
# STEP 3: Save config
config = {
    "vocab_size": 100,
    "hidden_size": 10,
    "num_hidden_layers": 1,
    "num_attention_heads": 1,
    "max_position_embeddings": 10
}
with open("model/config.json", "w") as f:
    json.dump(config, f, indent=2)
print("Saved config.json")

# -----------------------
# STEP 4: Save dummy tokenizer
with open("model/tokenizer.json", "w") as f:
    f.write('{}')
with open("model/tokenizer_config.json", "w") as f:
    json.dump({"do_lower_case": True}, f, indent=2)
print("Saved tokenizer.json and tokenizer_config.json")

print("\nAll files are ready in the 'model/' folder ✅")
