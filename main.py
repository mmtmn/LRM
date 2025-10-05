# lrm_conversational.py
import torch
import torch.nn as nn
import random
import os
from typing import List, Tuple

# -----------------------------
# Utilities for dataset loading
# -----------------------------
def load_triples(path: str) -> List[Tuple[str, str, str]]:
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples

def build_vocab(triples: List[Tuple[str, str, str]]):
    entities = sorted({h for h, _, t in triples} | {t for h, _, t in triples})
    relations = sorted({r for _, r, _ in triples})
    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}
    return ent2id, rel2id

def triples_to_ids(triples, ent2id, rel2id):
    triples_idx = []
    for h, r, t in triples:
        if h in ent2id and r in rel2id and t in ent2id:
            triples_idx.append((ent2id[h], rel2id[r], ent2id[t]))
    return triples_idx

def build_sequences(triples, seq_len=5):
    seqs = []
    for i in range(len(triples) - seq_len):
        seq = triples[i:i + seq_len]
        target = triples[i + seq_len][1]  # next relation
        seqs.append((seq, target))
    return seqs

# -----------------------------
# Relational Transformer Model
# -----------------------------
class RelationalBlock(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x

class LRMConversational(nn.Module):
    def __init__(self, n_entities, n_relations, emb_dim=256, depth=4, heads=4):
        super().__init__()
        self.entity_emb = nn.Embedding(n_entities, emb_dim)
        self.relation_emb = nn.Embedding(n_relations, emb_dim)
        self.encoder = nn.ModuleList([RelationalBlock(emb_dim, heads) for _ in range(depth)])
        self.proj = nn.Linear(emb_dim * 3, emb_dim)
        self.out_proj = nn.Linear(emb_dim, n_relations)

    def forward(self, seq):
        device = self.entity_emb.weight.device
        h_ids = torch.tensor([h for h, _, _ in seq], device=device)
        r_ids = torch.tensor([r for _, r, _ in seq], device=device)
        t_ids = torch.tensor([t for _, _, t in seq], device=device)

        h_vecs = self.entity_emb(h_ids)
        r_vecs = self.relation_emb(r_ids)
        t_vecs = self.entity_emb(t_ids)
        x = torch.cat([h_vecs, r_vecs, t_vecs], dim=-1)
        x = self.proj(x).unsqueeze(0)

        for blk in self.encoder:
            x = blk(x)

        return self.out_proj(x[:, -1, :])  # prediction for next relation

# -----------------------------
# Training
# -----------------------------
def train_lrm_conversational(data_dir, epochs=5, emb_dim=256, seq_len=5):
    train_path = os.path.join(data_dir, "train.txt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"No train.txt found in {data_dir}")

    triples = load_triples(train_path)
    ent2id, rel2id = build_vocab(triples)
    triples_idx = triples_to_ids(triples, ent2id, rel2id)
    dataset = build_sequences(triples_idx, seq_len=seq_len)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LRMConversational(len(ent2id), len(rel2id), emb_dim=emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Loaded {len(triples_idx):,} triples | {len(ent2id):,} entities | {len(rel2id):,} relations")

    for epoch in range(epochs):
        random.shuffle(dataset)
        total = 0
        for seq, next_rel in dataset:
            opt.zero_grad()
            logits = model(seq)  # [1, n_relations]
            target = torch.tensor([next_rel], device=device)
            loss = loss_fn(logits, target)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch:02d} | Loss {total/len(dataset):.4f}")

    # Quick sample inference
    seq, _ = random.choice(dataset)
    pred = model(seq)
    topk = torch.topk(pred, 5).indices.tolist()[0]
    print("\nContext:")
    for h, r, t in seq:
        print(f"({h}, {r}, {t})")
    print("\nTop-5 predicted next relations:")
    for k in topk:
        print(list(rel2id.keys())[list(rel2id.values()).index(k)])

    return model, ent2id, rel2id

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    # Choose dataset here:
    datasets = [
        "data/dialogre",      # conversational
        "data/FB15k-237/data",
        "data/wikidata5m",
        "data/WN18RR",
    ]

    # Loop through each dataset
    for ds in datasets:
        if not os.path.exists(ds):
            print(f"Skipping {ds} (not found)")
            continue
        print(f"\n=== Training on {ds} ===")
        train_lrm_conversational(ds, epochs=3, emb_dim=256)
