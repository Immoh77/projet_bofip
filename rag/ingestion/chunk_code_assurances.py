import json
import re
from uuid import uuid4
from pathlib import Path
import pdfplumber
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
import tiktoken
import logging
from rag.config import DOCUMENT_SOURCES

# === INITIALISATION ===
nltk.download("punkt")

logger = logging.getLogger(__name__)
tokenizer = tiktoken.get_encoding("cl100k_base")
tokenizer_fr = PunktSentenceTokenizer()

# === CONFIGURATION ===
source_cfg = DOCUMENT_SOURCES["code_assurances"]
PDF_PATH = Path(source_cfg["PDF_PATH"])
OUTPUT_BIG_CHUNKS = Path(source_cfg["OUTPUT_BIG_CHUNKS"])
OUTPUT_SMALL_CHUNKS = Path(source_cfg["OUTPUT_SMALL_CHUNKS"])
CHUNK_SIZE = source_cfg["CHUNK_SIZE"]
CHUNK_OVERLAP = source_cfg["CHUNK_OVERLAP"]

# === MOTIFS DE STRUCTURE ===
ARTICLE_PATTERN = re.compile(r"Article\s+[A-Z]?\s*\d{1,5}(?:-\d+)?", re.IGNORECASE)
TITLE_PATTERNS = {
    "livre": re.compile(r"Livre\s+(?:[IVXLCDM]+|Ier|IIe|IIIe|IVe|Ve)\s*:\s*(.+?)(?=\n(?:Titre|Chapitre|Article|Livre|\Z))", re.IGNORECASE | re.DOTALL),
    "titre": re.compile(r"Titre\s+(?:[IVXLCDM]+|Ier|IIe|IIIe|IVe|Ve)\s*:\s*(.+?)(?=\n(?:Chapitre|Article|Livre|Titre|\Z))", re.IGNORECASE | re.DOTALL),
    "chapitre": re.compile(r"Chapitre\s+(?:[IVXLCDM]+|Ier|IIe|IIIe|IVe|Ve)\s*:\s*(.+?)(?=\n(?:Section|Article|Chapitre|Livre|\Z))", re.IGNORECASE | re.DOTALL),
    "section": re.compile(r"Section\s+(?:[IVXLCDM]+|1(?:er)?|2e|3e|4e|5e)\s*:\s*(.+?)(?=\n(?:Article|Chapitre|Section|\Z))", re.IGNORECASE | re.DOTALL),
}


# === EXTRACTION DES STRUCTURES ===
def extract_structure_timeline(text):
    timeline = []
    for level, pattern in TITLE_PATTERNS.items():
        for match in pattern.finditer(text):
            title_text = re.sub(
                r"(Dernière modification le|Document généré le)\s+\d{1,2}\s+\w+\s+\d{4}",
                "",
                match.group(1).strip(),
                flags=re.IGNORECASE
            ).strip()
            if title_text:
                timeline.append({
                    "type": level,
                    "title": title_text,
                    "position": match.start()
                })
    return sorted(timeline, key=lambda x: x["position"])


def get_context_titles(position, timeline):
    context = {"livre": "", "titre": "", "chapitre": "", "section": ""}
    last_seen = {"livre": "", "titre": "", "chapitre": "", "section": ""}
    for item in timeline:
        if item["position"] > position:
            break
        last_seen[item["type"]] = item["title"]
    context.update(last_seen)
    return context


# === EXTRACTION DES ARTICLES ===
def extract_articles_from_pdf(pdf_path: Path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

    structure_timeline = extract_structure_timeline(text)
    initial_context = get_context_titles(0, structure_timeline)

    logger.info(f"📄 Texte extrait du PDF ({len(text)} caractères)")
    matches = list(ARTICLE_PATTERN.finditer(text))
    logger.info(f"🔍 Articles détectés : {len(matches)}")

    articles = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        titre = match.group(0).strip()
        article_id = titre

        contenu_brut = text[start:end].strip()
        contenu_brut = re.sub(
            r"(Dernière modification le|Document généré le)\s+\d{1,2}\s+\w+\s+\d{4}",
            "",
            contenu_brut,
            flags=re.IGNORECASE
        )

        context = get_context_titles(match.start(), structure_timeline)
        for key in ["livre", "titre", "chapitre", "section"]:
            if not context[key]:
                context[key] = initial_context[key]

        contenu_nettoye = "\n".join(
            line.strip() for line in contenu_brut.splitlines()
            if line.strip() and not re.match(r"^(Livre|Titre|Chapitre|Section)\s+", line.strip(), re.IGNORECASE)
        )

        context_lines = [
            f"{k.capitalize()} : {v}"
            for k, v in context.items() if v
        ]
        contenu_final = "\n".join(context_lines) + "\n\n" + contenu_nettoye

        articles.append({
            "titre_document": context["livre"],
            "titre_bloc": context["titre"],
            "division": context["chapitre"],
            "document_id": "",
            "permalien": article_id,
            "contenu": contenu_final
        })

    return articles


# === MÉTADONNÉES ===
def build_metadata(block, chunk_id=None, parent_chunk_id=None):
    metadata = {
        "base": "juridique",
        "source": "code_assurances",
        "titre_document": block.get("titre_document", ""),
        "titre_bloc": block.get("titre_bloc", ""),
        "division": block.get("division", ""),
        "document_id": block.get("document_id", ""),
        "permalien": block.get("permalien", ""),
    }
    if chunk_id:
        metadata["chunk_id"] = chunk_id
    if parent_chunk_id:
        metadata["parent_chunk_id"] = parent_chunk_id
    return metadata


# === CHUNKING PRINCIPAL ===
def split_with_overlap(blocks, max_tokens=800, overlap=50):
    chunks = []
    for block in blocks:
        paragraphs = block.get("contenu", "").split("\n")
        current_chunk, current_tokens = [], []
        chunk_index, i = 0, 0

        while i < len(paragraphs):
            para = paragraphs[i].strip()
            if not para:
                i += 1
                continue

            tokens_para = tokenizer.encode(para)
            para_len = len(tokens_para)

            if para_len > max_tokens:
                for j in range(0, para_len, max_tokens):
                    sub_tokens = tokens_para[j:j + max_tokens]
                    sub_text = tokenizer.decode(sub_tokens)
                    chunk_id = str(uuid4())
                    chunks.append({
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "contenu": sub_text.strip(),
                        "metadata": build_metadata(block, chunk_id=chunk_id)
                    })
                    chunk_index += 1
                i += 1
                continue

            total_tokens = sum(len(t) for t in current_tokens)
            if total_tokens + para_len > max_tokens:
                chunk_text = "\n".join(current_chunk)
                chunk_id = str(uuid4())
                chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "contenu": chunk_text.strip(),
                    "metadata": build_metadata(block, chunk_id=chunk_id)
                })
                chunk_index += 1

                n_tokens = 0
                j = len(current_tokens) - 1
                while j >= 0 and n_tokens < overlap:
                    n_tokens += len(current_tokens[j])
                    j -= 1
                current_chunk = current_chunk[j + 1:]
                current_tokens = current_tokens[j + 1:]

            current_chunk.append(para)
            current_tokens.append(tokens_para)
            i += 1

        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunk_id = str(uuid4())
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "contenu": chunk_text.strip(),
                "metadata": build_metadata(block, chunk_id=chunk_id)
            })
    return chunks


# === PETITS CHUNKS ===
def generate_small_chunks(big_chunks, max_tokens=100, min_tokens=30):
    small_chunks = []
    for chunk in big_chunks:
        content = chunk["contenu"]
        metadata = chunk["metadata"]

        prefix = "\n".join(
            filter(None, [
                metadata.get("titre_document", ""),
                metadata.get("titre_bloc", ""),
                metadata.get("division", "")
            ])
        ).strip()

        sentences = tokenizer_fr.tokenize(content)
        current, token_count, sub_index = [], 0, 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            tokens = tokenizer.encode(sent)
            t_len = len(tokens)

            if token_count + t_len > max_tokens and current:
                chunk_text = " ".join(current)
                if len(tokenizer.encode(chunk_text)) >= min_tokens:
                    chunk_id = f"{metadata['chunk_id']}__{sub_index}"
                    enriched = metadata.copy()
                    enriched["chunk_id"] = chunk_id
                    enriched["parent_chunk_id"] = metadata["chunk_id"]
                    small_chunks.append({
                        "id": chunk_id,
                        "parent_chunk_id": metadata["chunk_id"],
                        "small_index": sub_index,
                        "contenu": f"{prefix}\n\n{chunk_text}".strip(),
                        "metadata": enriched,
                    })
                    sub_index += 1
                current, token_count = [sent], t_len
            else:
                current.append(sent)
                token_count += t_len

        if current:
            chunk_text = " ".join(current)
            if len(tokenizer.encode(chunk_text)) >= min_tokens:
                chunk_id = f"{metadata['chunk_id']}__{sub_index}"
                enriched = metadata.copy()
                enriched["chunk_id"] = chunk_id
                enriched["parent_chunk_id"] = metadata["chunk_id"]
                small_chunks.append({
                    "id": chunk_id,
                    "parent_chunk_id": metadata["chunk_id"],
                    "small_index": sub_index,
                    "contenu": f"{prefix}\n\n{chunk_text}".strip(),
                    "metadata": enriched,
                })
    return small_chunks


# === PIPELINE PRINCIPAL ===
def main():
    logger.info("==== Chunking du Code des assurances ====")
    logger.info(f"📘 Lecture du PDF : {PDF_PATH}")

    articles = extract_articles_from_pdf(PDF_PATH)
    logger.info(f"✅ Articles extraits : {len(articles)}")

    big_chunks = split_with_overlap(articles, max_tokens=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    logger.info(f"✅ Gros chunks générés : {len(big_chunks)}")

    OUTPUT_BIG_CHUNKS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_BIG_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(big_chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Sauvegarde : {OUTPUT_BIG_CHUNKS}")

    small_chunks = generate_small_chunks(big_chunks)
    with open(OUTPUT_SMALL_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(small_chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Petits chunks sauvegardés : {OUTPUT_SMALL_CHUNKS}")
    logger.info("🎉 Chunking terminé avec succès.")


if __name__ == "__main__":
    main()
