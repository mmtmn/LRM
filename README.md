# Large Relational Model (LRM)

This project implements **LRM**, a hybrid graph–embedding model that predicts the next relation in a sequence of knowledge graph triples.  
It supports training on standard benchmarks such as **FB15k-237** and **WN18RR**.

---

## Setup

### 1. Create the data folder
```bash
mkdir -p data && cd data
````

### 2. Download FB15k-237

Clone the dataset from Hugging Face:

```bash
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/KGraph/FB15k-237
# Optional: clone only pointer files (no large data)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/KGraph/FB15k-237
```

After cloning, the train/valid/test files will be in:

```
data/FB15k-237/data/
```

### 3. Download WN18RR

```bash
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/VLyb/WN18RR
# Optional: clone only pointer files (no large data)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/VLyb/WN18RR
```

---

## Running

From the project root:

```bash
python3 main.py
```

By default, it trains on `data/FB15k-237/data`.
To switch to WN18RR, edit the last line of `main.py`:

```python
model, ent2id, rel2id = train_lrm("data/WN18RR", epochs=20)
```

---

## Requirements

* Python 3.10+
* PyTorch
* NumPy
* Git LFS

```bash
pip install torch numpy
```

---

## Folder Structure

```
LRM/
 ├─ main.py
 ├─ README.md
 └─ data/
     ├─ FB15k-237/
     │   └─ data/
     │       ├─ train.txt
     │       ├─ valid.txt
     │       └─ test.txt
     └─ WN18RR/
         ├─ train.csv
         ├─ valid.csv
         └─ test.csv
```


