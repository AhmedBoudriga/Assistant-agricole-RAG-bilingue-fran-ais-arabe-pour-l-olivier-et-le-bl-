"""Extraction du texte des PDF du corpus (texte natif ou OCR).

Sortie : data/processed/<doc_id>/pages.jsonl (une ligne JSON par page).

Usage :
    python -m src.ingestion.extract --doc ble_fr_002
    python -m src.ingestion.extract --doc ble_fr_002 --pages 11-12   # test
    python -m src.ingestion.extract --doc ble_fr_002 --force         # refait tout
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pymupdf
import pytesseract
import yaml
from PIL import Image
from tqdm import tqdm

LANG_TESSERACT = {"fr": "fra", "ar": "ara", "en": "eng"}
MIN_NATIVE_CHARS = 50  # en dessous, la page est considérée comme une image


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_pages(spec: str | None, n_pages: int) -> list[int]:
    """'11-12' -> [10, 11] (index 0-based). None -> toutes les pages."""
    if not spec:
        return list(range(n_pages))
    start, _, end = spec.partition("-")
    start = int(start)
    end = int(end) if end else start
    return list(range(start - 1, min(end, n_pages)))


def extract_page(page, lang: str, dpi: int) -> tuple[str, str]:
    """Retourne (texte, méthode) où méthode vaut 'text' ou 'ocr'."""
    native = page.get_text().strip()
    if len(native) >= MIN_NATIVE_CHARS:
        return native, "text"
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img, lang=lang).strip(), "ocr"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", required=True, help="doc_id du registre")
    parser.add_argument("--pages", help="ex. 11-12 (numérotation à partir de 1)")
    parser.add_argument("--force", action="store_true", help="refait l'extraction")
    args = parser.parse_args()

    cfg = load_config()
    dpi = cfg.get("ocr", {}).get("dpi", 300)

    registre = pd.read_csv("data/sources.csv")
    lignes = registre[registre["doc_id"] == args.doc]
    if lignes.empty:
        raise SystemExit(f"doc_id inconnu dans data/sources.csv : {args.doc}")
    ligne = lignes.iloc[0]

    lang = LANG_TESSERACT.get(str(ligne["langue"]).strip(), "fra")
    pdf_path = Path(ligne["fichier_local"])
    if not pdf_path.exists():
        raise SystemExit(f"Fichier introuvable : {pdf_path}")

    out_dir = Path(cfg["paths"]["processed"]) / args.doc
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "pages.jsonl"
    if args.force and out_file.exists():
        out_file.unlink()

    deja_faites = set()
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            deja_faites = {json.loads(l)["page"] for l in f if l.strip()}

    doc = pymupdf.open(pdf_path)
    indices = [i for i in parse_pages(args.pages, len(doc)) if (i + 1) not in deja_faites]
    print(f"{args.doc} : {len(doc)} pages, {len(indices)} à traiter (langue OCR : {lang}, dpi : {dpi})")

    debut = time.time()
    with open(out_file, "a", encoding="utf-8") as f:
        for i in tqdm(indices, unit="page"):
            texte, methode = extract_page(doc[i], lang, dpi)
            f.write(json.dumps({
                "doc_id": args.doc,
                "page": i + 1,
                "methode": methode,
                "n_chars": len(texte),
                "texte": texte,
            }, ensure_ascii=False) + "\n")
            f.flush()  # rien n'est perdu si on interrompt

    print(f"Terminé en {time.time() - debut:.0f} s -> {out_file}")


if __name__ == "__main__":
    main()