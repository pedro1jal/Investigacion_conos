"""Exporta embeddings de datasets de clasificación textual a CSV y JSON.

Instalación:
    pip install -U datasets sentence-transformers gensim numpy tqdm torch

Ejemplo de prueba:
    python exportar_embeddings_texto.py --max-por-split 100

Para procesar todos los ejemplos (puede ocupar decenas de GB):
    python exportar_embeddings_texto.py --max-por-split 0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_id: str
    n_classes: int
    text_column: str = "text"
    label_column: str = "label"
    config: str | None = None


DATASETS = [
    DatasetSpec("AG_News", "fancyzhx/ag_news", 4),
    DatasetSpec("DBpedia_14", "fancyzhx/dbpedia_14", 14, text_column="content"),
    DatasetSpec("BBC_News", "SetFit/bbc-news", 5),
    DatasetSpec("20_Newsgroups", "SetFit/20_newsgroups", 20),
    DatasetSpec("Emotion_6", "dair-ai/emotion", 6, config="split"),
    DatasetSpec("SST_5", "SetFit/sst5", 5),
]

MODELS = [
    "word2vec-google-news-300",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]

TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def slug_model(model_id: str) -> str:
    if model_id == "word2vec-google-news-300":
        return model_id
    return model_id.rsplit("/", 1)[-1]


def create_encoder(model_id: str, device: str | None):
    if model_id == "word2vec-google-news-300":
        import gensim.downloader as api
        print("Cargando Google News Word2Vec (~1.6 GB de descarga inicial)...", flush=True)
        return api.load(model_id)

    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id, device=device) if device else SentenceTransformer(model_id)


def encode_texts(model_id: str, model: Any, texts: list[str], normalize: bool) -> np.ndarray:
    if model_id != "word2vec-google-news-300":
        vectors = model.encode(
            texts,
            batch_size=len(texts),
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    # Word2Vec proporciona vectores para palabras, no para documentos.
    # Representación del documento: promedio de palabras presentes en su vocabulario.
    dim = model.vector_size
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        words = TOKEN_PATTERN.findall(text.lower())
        # Google News distingue mayúsculas/minúsculas: fallback al token original.
        words_original = TOKEN_PATTERN.findall(text)
        selected = [lo if lo in model else orig for lo, orig in zip(words, words_original)]
        selected = [word for word in selected if word in model]
        if selected:
            out[i] = np.mean(model[selected], axis=0, dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out /= np.maximum(norms, 1e-12)
    return out


def label_names_for(split_dataset: Any, label_column: str) -> list[str] | None:
    feature = (getattr(split_dataset, "features", None) or {}).get(label_column)
    names = getattr(feature, "names", None)
    return list(names) if names else None


def get_label_name(label: Any, names: list[str] | None) -> str:
    if label is None:
        return ""
    if names is not None:
        try:
            idx = int(label)
            if 0 <= idx < len(names):
                return names[idx]
        except (TypeError, ValueError):
            pass
    return str(label)


def row_text(example: dict, spec: DatasetSpec) -> str:
    value = example.get(spec.text_column)
    if value is None:
        raise KeyError(f"No existe columna de texto '{spec.text_column}'. Disponibles: {list(example)}")
    if isinstance(value, list):
        return " ".join(map(str, value))
    return str(value)


def iter_batches(data: Any, batch_size: int, max_rows: int | None):
    batch = []
    for idx, example in enumerate(data):
        if max_rows is not None and idx >= max_rows:
            break
        batch.append((idx, example))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def export_split(
    ds: Any, split: str, spec: DatasetSpec, model_id: str, model: Any,
    output_dir: Path, batch_size: int, max_rows: int | None,
    normalize: bool, overwrite: bool, save_text: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split}.csv"
    meta_path = output_dir / f"{split}.json"
    if path.exists() and meta_path.exists() and not overwrite:
        print(f"  Omitido (ya existe): {path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    names = label_names_for(ds, spec.label_column)
    # Se guardan las columnas originales no vectorizadas aparte del texto principal.
    original_columns = list(getattr(ds, "column_names", []) or [])
    extras = [c for c in original_columns if c not in {spec.text_column, spec.label_column}]
    reserved = {"id", "dataset", "split", "label", "label_name", "text", "text_length", "n_tokens"}
    extras = [c for c in extras if c not in reserved and not c.startswith("emb_")]

    tmp = path.with_suffix(".csv.part")
    count = 0
    dim = None
    zero_vectors = 0
    distribution: dict[str, int] = {}

    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for batch in tqdm(iter_batches(ds, batch_size, max_rows), desc=f"{spec.name}/{split}/{slug_model(model_id)}", unit="batch"):
                texts = [row_text(example, spec) for _, example in batch]
                X = encode_texts(model_id, model, texts, normalize)
                if X.ndim != 2 or X.shape[0] != len(batch):
                    raise ValueError(f"Forma incorrecta de embeddings: {X.shape}")

                if dim is None:
                    dim = X.shape[1]
                    columns = ["id", "dataset", "split", "label", "label_name", "text_length", "n_tokens"]
                    if save_text:
                        columns.append("text")
                    columns += extras
                    columns += [f"emb_{j:04d}" for j in range(dim)]
                    writer.writerow(columns)

                zero_vectors += int(np.sum(np.linalg.norm(X, axis=1) < 1e-10))
                for (source_idx, example), txt, vector in zip(batch, texts, X):
                    label = example.get(spec.label_column)
                    if label is None:
                        raise KeyError(f"Falta etiqueta '{spec.label_column}' en {spec.name}/{split}")
                    label_name = get_label_name(label, names)
                    distribution[str(label)] = distribution.get(str(label), 0) + 1
                    row = [
                        f"{spec.name}:{split}:{source_idx}", spec.name, split,
                        label, label_name, len(txt), len(txt.split()),
                    ]
                    if save_text:
                        row.append(txt)
                    for key in extras:
                        value = example.get(key)
                        row.append(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
                    row.extend(float(x) for x in vector)
                    writer.writerow(row)
                    count += 1

            if dim is None:  # partición sin datos
                dim = int(getattr(model, "vector_size", 0) or getattr(model, "get_sentence_embedding_dimension", lambda: 0)())
                columns = ["id", "dataset", "split", "label", "label_name", "text_length", "n_tokens"]
                if save_text:
                    columns.append("text")
                columns += extras + [f"emb_{j:04d}" for j in range(dim)]
                writer.writerow(columns)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    metadata = {
        "dataset": asdict(spec),
        "split": split,
        "model": model_id,
        "csv": path.name,
        "rows": count,
        "embedding_dimensions": dim,
        "normalized_l2": normalize,
        "zero_vectors": zero_vectors,
        "label_names": names,
        "label_counts": distribution,
        "saved_original_columns": ([spec.text_column] if save_text else []) + extras,
        "text_truncation": "SentenceTransformer default tokenizer limits; original text retained separately when enabled",
        "word2vec_aggregation": "unweighted mean of in-vocabulary word vectors; zero if no words in vocabulary" if model_id == "word2vec-google-news-300" else None,
        "max_rows_per_split": max_rows,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Guardado: {path} | X=({count}, {dim}), ceros={zero_vectors}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", default="embeddings_exportados")
    parser.add_argument("--max-por-split", type=int, default=5000, help="0 = todas las filas; predeterminado 5000 por partición")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--datasets", nargs="+", default=[d.name for d in DATASETS], choices=[d.name for d in DATASETS])
    parser.add_argument("--modelos", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--device", default=None, help="Ej. cuda, cpu, o automático")
    parser.add_argument("--sin-normalizar", action="store_true")
    parser.add_argument("--sin-texto", action="store_true", help="No guardar texto completo en el CSV")
    parser.add_argument("--sobrescribir", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_por_split < 0:
        parser.error("batch-size debe ser >= 1 y max-por-split >= 0")

    specs = [d for d in DATASETS if d.name in args.datasets]
    root = Path(args.salida)
    root.mkdir(parents=True, exist_ok=True)
    limit = args.max_por_split or None
    index = []

    # Un modelo a la vez: evita mantener los tres en RAM/VRAM simultáneamente.
    for model_id in args.modelos:
        print(f"\n========== MODELO: {model_id} ==========", flush=True)
        model = create_encoder(model_id, args.device)
        for spec in specs:
            print(f"\nDataset: {spec.name} ({spec.hf_id})", flush=True)
            try:
                # Streaming permite procesar DBpedia sin cargar el dataset completo.
                stream = load_dataset(spec.hf_id, name=spec.config, streaming=True)
                splits = list(stream.keys()) if hasattr(stream, "keys") else ["train"]
                for split in splits:
                    data = stream[split] if hasattr(stream, "keys") else stream
                    info = export_split(
                        data, split, spec, model_id, model,
                        root / spec.name / slug_model(model_id),
                        args.batch_size, limit, not args.sin_normalizar,
                        args.sobrescribir, not args.sin_texto,
                    )
                    index.append({
                        "dataset": spec.name, "model": model_id, "split": split,
                        "rows": info["rows"], "dimensions": info["embedding_dimensions"],
                        "csv": str(Path(spec.name) / slug_model(model_id) / f"{split}.csv"),
                    })
            except Exception:
                logging.exception("Error procesando dataset %s con modelo %s", spec.name, model_id)
                raise
        del model

    (root / "indice.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nProceso completado. Índice: {root / 'indice.json'}")


if __name__ == "__main__":
    main()
