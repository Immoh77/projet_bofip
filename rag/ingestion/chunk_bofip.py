import json
from bs4 import BeautifulSoup
from uuid import uuid4
import tiktoken
import os
from rag.config import (
    SOURCE_FILE,
    OUTPUT_BIG_CHUNKS,
    OUTPUT_SMALL_CHUNKS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    ALLOWED_SERIES,
    EXCLUDED_DOCUMENT_PREFIXES,
)

# === INITIALISATION ===
tokenizer = tiktoken.get_encoding("cl100k_base")  # Tokenizer OpenAI

# === FILTRAGE DES DOCUMENTS ===
def filter_documents_by_series(data):
    """Filtre les documents selon la série autorisée et les préfixes exclus."""
    return [
        doc for doc in data
        if isinstance(doc.get("serie"), str)
        and doc["serie"].strip().upper() in ALLOWED_SERIES
        and not any(str(doc.get("identifiant_juridique", "")).startswith(prefix) for prefix in EXCLUDED_DOCUMENT_PREFIXES)
    ]

# === EXTRACTION DES BLOCS TEXTUELS STRUCTURÉS ===
def extract_blocks_from_html(documents):
    """Extrait des blocs de texte structurés depuis le HTML."""
    blocks = []
    for doc in documents:
        html = doc.get("contenu_html")
        if not html:
            continue

        soup = BeautifulSoup(html, "html.parser")
        block_title = ""
        block_content = []
        paragraph_number = ""

        for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
            text = el.get_text()
            if text.strip().isdigit():
                paragraph_number = text.strip()
                continue

            if paragraph_number:
                text = f"§ {paragraph_number} - {text.strip()}"
                paragraph_number = ""
            else:
                text = text.strip()

            if not text:
                continue

            tag = el.name.lower()
            if tag in ["h1", "h2", "h3"]:
                block_title = text
                continue

            block_content.append(text)

        if block_content:
            blocks.append({
                "titre_document": doc.get("titre", ""),
                "titre_bloc": block_title,
                "serie": doc.get("serie", ""),
                "division": doc.get("division", ""),
                "document_id": doc.get("identifiant_juridique", ""),
                "permalien": doc.get("permalien", ""),
                "contenu": "\n".join(block_content),
            })

    return blocks

# === CHUNKING AVEC OVERLAP ===
def split_with_overlap(blocks, max_tokens=700, overlap=100):
    """Découpe les blocs en chunks avec chevauchement (overlap)."""
    refined_chunks = []

    for block in blocks:
        paragraphs = block.get("contenu", "").split("\n")
        current_chunk = []
        current_tokens = []
        chunk_index = 0
        i = 0

        while i < len(paragraphs):
            para = paragraphs[i].strip()
            if not para:
                i += 1
                continue

            tokens_para = tokenizer.encode(para)
            para_token_len = len(tokens_para)

            # Si un paragraphe est trop long
            if para_token_len > max_tokens:
                for j in range(0, para_token_len, max_tokens):
                    sub_tokens = tokens_para[j:j + max_tokens]
                    sub_text = tokenizer.decode(sub_tokens)
                    chunk_id = str(uuid4())
                    metadata = build_metadata(block, chunk_id=chunk_id)
                    refined_chunks.append({
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "contenu": sub_text.strip(),
                        "metadata": metadata,
                    })
                    chunk_index += 1
                i += 1
                continue

            total_tokens = sum(len(t) for t in current_tokens)
            if total_tokens + para_token_len > max_tokens:
                chunk_text = "\n".join(current_chunk)
                chunk_id = str(uuid4())
                metadata = build_metadata(block, chunk_id=chunk_id)
                refined_chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "contenu": chunk_text.strip(),
                    "metadata": metadata,
                })
                chunk_index += 1

                # Overlap
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

        # Dernier chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunk_id = str(uuid4())
            metadata = build_metadata(block, chunk_id=chunk_id)
            refined_chunks.append({
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "contenu": chunk_text.strip(),
                "metadata": metadata,
            })

    return refined_chunks

# === MÉTADONNÉES ===
def build_metadata(block, chunk_id=None, parent_chunk_id=None):
    metadata = {
        "base": "fiscal",
        "source": "bofip",
        "titre_document": block["titre_document"],
        "titre_bloc": block["titre_bloc"],
        "serie": block["serie"],
        "division": block["division"],
        "document_id": block["document_id"],
        "permalien": block["permalien"],
    }
    if chunk_id:
        metadata["chunk_id"] = chunk_id
    if parent_chunk_id:
        metadata["parent_chunk_id"] = parent_chunk_id
    return metadata

# === PETITS CHUNKS POUR L’EMBEDDING ===
def generate_small_chunks(input_path, output_path, max_tokens=100):
    """Découpe les gros chunks en sous-chunks plus petits (embedding)."""

    def deduplicate_chunks(chunks):
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            key = (chunk["contenu"].strip(), chunk["metadata"]["document_id"])
            if key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)
        return unique_chunks

    with open(input_path, "r", encoding="utf-8") as f:
        big_chunks = json.load(f)

    small_chunks = []
    for chunk in big_chunks:
        content = chunk["contenu"]
        parent_id = chunk["chunk_id"]
        metadata = chunk["metadata"]

        prefix = f"{metadata['titre_document']} - {metadata['titre_bloc']}".strip()
        phrases = content.split("\n")
        current_text = []
        current_tokens = 0
        sub_index = 0

        for phrase in phrases:
            phrase = phrase.strip()
            if not phrase:
                continue

            tokens = tokenizer.encode(phrase)
            token_len = len(tokens)

            if current_tokens + token_len > max_tokens and current_text:
                chunk_id = f"{parent_id}__{sub_index}"
                chunk_metadata = build_metadata(metadata, chunk_id=chunk_id, parent_chunk_id=parent_id)
                small_chunks.append({
                    "id": chunk_id,
                    "parent_chunk_id": parent_id,
                    "small_index": sub_index,
                    "contenu": f"{prefix}\n" + "\n".join(current_text).strip(),
                    "metadata": chunk_metadata,
                })
                sub_index += 1
                current_text = []
                current_tokens = 0

            current_text.append(phrase)
            current_tokens += token_len

        if current_text:
            chunk_id = f"{parent_id}__{sub_index}"
            chunk_metadata = build_metadata(metadata, chunk_id=chunk_id, parent_chunk_id=parent_id)
            small_chunks.append({
                "id": chunk_id,
                "parent_chunk_id": parent_id,
                "small_index": sub_index,
                "contenu": f"{prefix}\n" + "\n".join(current_text).strip(),
                "metadata": chunk_metadata,
            })

    small_chunks = deduplicate_chunks(small_chunks)

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(small_chunks, f_out, indent=2, ensure_ascii=False)


# === PIPELINE PRINCIPAL ===
def main():
    print("🔄 Loading source file...")
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📄 Total documents: {len(data)}")

    print("🔍 Filtering by allowed series...")
    filtered_docs = filter_documents_by_series(data)
    print(f"✅ Documents retained: {len(filtered_docs)}")

    print("🧱 Extracting content blocks...")
    blocks = extract_blocks_from_html(filtered_docs)
    print(f"✅ Blocks extracted: {len(blocks)}")

    print("✂️ Creating big chunks...")
    big_chunks = split_with_overlap(blocks, max_tokens=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"📦 Big chunks created: {len(big_chunks)}")

    print("💾 Saving big chunks...")
    with open(OUTPUT_BIG_CHUNKS, "w", encoding="utf-8") as f_out:
        json.dump(big_chunks, f_out, indent=2, ensure_ascii=False)

    print("✂️ Creating small chunks...")
    generate_small_chunks(
        input_path=OUTPUT_BIG_CHUNKS,
        output_path=OUTPUT_SMALL_CHUNKS,
        max_tokens=100,
    )

if __name__ == "__main__":
    main()
