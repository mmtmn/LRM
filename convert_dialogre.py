import json, os

in_path = "data/dialogre/train.json"
out_path = "data/dialogre/train.txt"

with open(in_path, "r", encoding="utf-8") as f:
    data = json.load(f)

triples = []
for dialog in data:
    if not isinstance(dialog, list) or len(dialog) < 2:
        continue
    relations = dialog[1]
    if not isinstance(relations, list):
        continue
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        head = rel.get("x") or ""
        tail = rel.get("y") or ""
        # 'r' is a list like ['per:employee_of']
        r_list = rel.get("r")
        relation = r_list[0] if isinstance(r_list, list) and len(r_list) > 0 else str(r_list)
        if head and relation and tail:
            triples.append((head.strip(), relation.strip(), tail.strip()))

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    for h, r, t in triples:
        f.write(f"{h}\t{r}\t{t}\n")

print(f"Wrote {len(triples)} clean triples to {out_path}")
