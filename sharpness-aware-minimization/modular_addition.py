import torch, math, random
from torch.utils.data import Dataset, DataLoader

class ModAddDataset(Dataset):
    """
    Each example: input seq [a, b, CLS], label = (a+b) mod p.
    Tokens: 0..p-1 for numbers, p is CLS.
    """
    def __init__(self, p=97, split="train", train_frac=0.2, seed=0):
        super().__init__()
        rng = random.Random(seed)
        all_pairs = [(a, b) for a in range(p) for b in range(p)]
        rng.shuffle(all_pairs)
        n_train = int(train_frac * len(all_pairs))
        self.p = p
        self.cls = p
        if split == "train":
            self.pairs = all_pairs[:n_train]
        else:
            self.pairs = all_pairs[n_train:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        x = torch.tensor([a, b, self.cls], dtype=torch.long)   # (L=3)
        y = torch.tensor((a + b) % self.p, dtype=torch.long)
        return x, y

def modadd_loaders(p=97, train_frac=0.2, batch_size=512, seed=0):
    ds_tr = ModAddDataset(p=p, split="train", train_frac=train_frac, seed=seed)
    ds_te = ModAddDataset(p=p, split="test",  train_frac=train_frac, seed=seed)
    tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=True)
    te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0)
    return ds_tr, ds_te, tr, te