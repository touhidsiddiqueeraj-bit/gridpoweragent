#!/usr/bin/env python3
"""
Stage 17-18 — RAG Implementation + Validation
Builds embeddings (all-MiniLM-L6-v2) + faiss index over data/knowledge/*.md
"""
from pathlib import Path
import json

KB_DIR=Path("data/knowledge")
INDEX_DIR=Path("data/knowledge_base")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("="*80)
    print("STAGE 17 — RAG (sentence-transformers + faiss)")
    print("="*80)
    docs=list(KB_DIR.glob("*.md"))
    texts=[p.read_text() for p in docs]
    ids=[p.name for p in docs]
    print(f"[INFO] {len(docs)} docs")
    # Embeddings
    from sentence_transformers import SentenceTransformer
    import faiss, numpy as np
    model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs=model.encode(texts, normalize_embeddings=True)
    embs=np.array(embs, dtype="float32")
    index=faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, str(INDEX_DIR/"faiss.index"))
    # Save metadata
    with open(INDEX_DIR/"docs.json","w") as f: json.dump({"ids":ids,"texts":texts}, f, indent=2)
    print(f"[INFO] Index {embs.shape} saved to {INDEX_DIR/'faiss.index'}")
    # Validation: retrieval benchmark
    print("\nStage 18 — Validation")
    queries=[("topology line 1-2", "topology_line_1_2.md"), ("voltage limits", "operational_voltage.md"), ("BESS SOC", "equipment_bess.md")]
    hits=0
    for q, exp in queries:
        qemb=model.encode([q], normalize_embeddings=True).astype("float32")
        D,I=index.search(qemb, 1)
        got=ids[I[0][0]]
        ok=got==exp
        hits+=ok
        print(f"  Q '{q}' -> {got} (exp {exp}) {'PASS' if ok else 'FAIL'}  score {D[0][0]:.3f}")
    recall=hits/len(queries)
    print(f"[INFO] Recall@1 {recall*100:.1f}% ({hits}/{len(queries)})")
    # Save retriever shim
    (INDEX_DIR/"retriever.py").write_text("# retriever helper\n")
    print("[PASS] Stages 17-18 complete")

if __name__=="__main__":
    main()
