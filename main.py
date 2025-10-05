import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os

# ---------- Data loading ----------
def load_triples(path):
    triples = []
    with open(path, "r") as f:
        for line in f:
            h, r, t = line.strip().split("\t")
            triples.append((h, r, t))
    return triples

def build_vocab(triples):
    ents = sorted({h for h,_,t in triples} | {t for h,_,t in triples})
    rels = sorted({r for _,r,_ in triples})
    ent2id = {e:i for i,e in enumerate(ents)}
    rel2id = {r:i for i,r in enumerate(rels)}
    return ent2id, rel2id

def triples_to_ids(triples, ent2id, rel2id):
    return [(ent2id[h], rel2id[r], ent2id[t]) for h,r,t in triples]

# ---------- Graph Memory ----------
class GraphMemory(nn.Module):
    def __init__(self, n_entities, emb_dim):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(n_entities, emb_dim) * 0.1)

    def message_pass(self, triples, rel_emb):
        h_idx, r_idx, t_idx = triples[:,0], triples[:,1], triples[:,2]
        h_vec, r_vec, t_vec = self.memory[h_idx], rel_emb(r_idx), self.memory[t_idx]
        msg = h_vec + r_vec - t_vec
        self.memory[h_idx] = 0.9*self.memory[h_idx] + 0.1*msg

# ---------- Model ----------
class LRMv2(nn.Module):
    def __init__(self, n_entities, n_relations, emb_dim=128):
        super().__init__()
        self.entity_mem = GraphMemory(n_entities, emb_dim)
        self.relation_emb = nn.Embedding(n_relations, emb_dim)
        self.temporal_rnn = nn.GRU(emb_dim*3, emb_dim, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, n_relations)
        )

    def forward(self, seq):
        seq_vecs = []
        for h, r, t in seq:
            h_vec = self.entity_mem.memory[h]
            r_vec = self.relation_emb(torch.tensor(r, device=h_vec.device))
            t_vec = self.entity_mem.memory[t]
            seq_vecs.append(torch.cat([h_vec, r_vec, t_vec]))
        seq_tensor = torch.stack(seq_vecs).unsqueeze(0)
        _, hidden = self.temporal_rnn(seq_tensor)
        return self.predictor(hidden.squeeze(0))

# ---------- Training ----------
def build_sequences(triples, seq_len=3):
    seqs = []
    for i in range(len(triples)-seq_len):
        seq = triples[i:i+seq_len]
        next_rel = triples[i+seq_len][1]
        seqs.append((seq, next_rel))
    return seqs

def train_lrm(dataset_dir="data/FB15k-237", epochs=50, emb_dim=128):
    train = load_triples(os.path.join(dataset_dir,"train.txt"))
    ent2id, rel2id = build_vocab(train)
    triples = triples_to_ids(train, ent2id, rel2id)
    dataset = build_sequences(triples)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LRMv2(len(ent2id), len(rel2id), emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total = 0
        random.shuffle(dataset)
        for seq, next_rel in dataset:
            opt.zero_grad()
            pred = model(seq)
            target = torch.tensor([next_rel], device=device)
            loss = criterion(pred.unsqueeze(0), target)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch:03d} | Loss: {total/len(dataset):.4f}")
    return model, ent2id, rel2id

# ---------- Run ----------
if __name__ == "__main__":
    model, ent2id, rel2id = train_lrm("data/FB15k-237", epochs=20)
    print("Training complete.")
