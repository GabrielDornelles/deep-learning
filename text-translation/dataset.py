from datasets import load_dataset
from typing import Tuple


class TatoebaDataset:

    def __init__(self, split: str = "train") -> None:
        self.dataset = load_dataset("tatoeba", lang1="en", lang2="pt")["train"]
        #size = int(len(self.dataset) * 0.85)
        if split == "train":
            self.dataset = self.dataset[:30000] #[:200000]
        if split == "val":
            self.dataset = self.dataset[30000:31000] # [200000:210000]

    def __len__(self) -> int:
        return len(self.dataset["translation"])

    def __getitem__(self, idx) -> Tuple[str,str]:
        item = self.dataset["translation"][idx]
        return (item["en"], item["pt"])
