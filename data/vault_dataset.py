import json
from torch.utils.data import Dataset

class VaultDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if "text" in data and len(data["text"]) > 50:
                    self.samples.append(data["text"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
