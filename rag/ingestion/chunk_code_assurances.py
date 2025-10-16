import json
import re
from uuid import uuid4
from pathlib import Path
import pdfplumber
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
import tiktoken
from rag.config import DOCUMENT_SOURCES

nltk.download("punkt")

# === CONFIGURATION ===
source_cfg = DOCUMENT_SOURCES["code_assurances"]
PDF_PATH = source_cfg["PDF_PATH"]
OUTPUT_BIG_CHUNKS = source_cfg["OUTPUT_BIG_CHUNKS"]
OUTPUT_SMALL_CHUNKS = source_cfg["OUTPUT_SMALL_CHUNKS"]
CHUNK_SIZE = source_cfg["CHUNK_SIZE"]
CHUNK_OVERLAP = source_cfg["CHUNK_OVERLAP"]

tokenizer = tiktoken.get_encoding("cl100k_base")
tokenizer_fr = PunktSentenceTokenizer()

ARTICLE_PATTERN = re.compile(r"Article\s+[A-Z]?\s*\d{1,5}(?:-\d+)?", re.IGNORECASE)

TITLE_PATTERNS = {
    "livre": re.compile(r"Livre\s+(?:[IVXLCDM]+|Ier|IIe|IIIe|IVe|Ve|VIe|VIIe|VIIIe|IXe|Xe)\s*:\s*(.+?)(?=\n(?:Titre|Chapitre|Article|Livre|\Z))", re.IGNORECASE | re.DOTALL),
    "titre": re.compile(r"Titre\s+(?:[IVXLCDM]+|Ier|IIe|IIIe|IVe|Ve|VIe|VIIe|VIIIe|IXe|Xe)\s*:\s*(.+?)(?=\n(?:Chapitre|Article|Livre|Titre|\Z))", re.IGNORECASE | re.DOTALL),
    "chapitre": re.compile(
        r"Chapitre\s+(?:[IVXLCDM]+|Ier|IIe|IIIe|IVe|Ve|VIe|VIIe|VIIIe|IXe|Xe)\s*:\s*(.+?)(?=\n(?:Section|Article|Chapitre|Livre|Titre|\Z))",
        re.IGNORECASE | re.DOTALL
    ),
    "section": re.compile(
        r"Section\s+(?:[IVXLCDM]+|1(?:er)?|2e|3e|4e|5e|6e|7e|8e|9e|10e?)\s*:\s*(.+?)(?=\n(?:Article|Chapitre|Section|Livre|Titre|\Z))",
    re.IGNORECASE | re.DOTALL
    )
}

def extract_structure_timeline(text):
    timeline = []
    for level, pattern in TITLE_PATTERNS.items():
        for match in pattern.finditer(text):
            # Ne pas exiger que le titre soit suivi de quelque chose
            title_text = re.sub(
                r"(Dernière modification le|Document généré le)\s+\d{1,2}\s+\w+\s+\d{4}(.*?)?(?=\n|$)",
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
    # Ajouter des valeurs par défaut si aucun titre trouvé
    return sorted(timeline, key=lambda x: x["position"])

def get_context_titles(position, timeline):
    context = {"livre": "", "titre": "", "chapitre": "","section" : ""}
    last_seen = {"livre": "", "titre": "", "chapitre": "", "section" : ""}

    for item in timeline:
        if item["position"] > position:
            break
        last_seen[item["type"]] = item["title"]

    context.update(last_seen)
    return context

def extract_articles_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

    structure_timeline = extract_structure_timeline(text)
    initial_context = get_context_titles(0, structure_timeline)

    print("Aperçu du texte brut extrait :")
    print(text[:1000])

    matches = list(ARTICLE_PATTERN.finditer(text))
    for m in matches[:5]:
        print(f"Article détecté : {m.group(0)} à la position {m.start()}")
    print(f">> Articles détectés : {len(matches)}")

    articles = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        titre = match.group(0).strip()
        article_id = titre

        contenu_brut = text[start:end].strip()

        # Suppression des mentions automatiques de mise à jour/génération
        contenu_brut = re.sub(
            r"(Dernière modification le|Document généré le)\s+\d{1,2}\s+\w+\s+\d{4}(.*?)?(?=\n|$)",
            "",
            contenu_brut,
            flags=re.IGNORECASE
        )

        # Obtenir le contexte de structure (avec section)
        context = get_context_titles(match.start(), structure_timeline)

        # Fallback vers le contexte initial si vide
        for key in ["livre", "titre", "chapitre", "section"]:
            if not context[key]:
                context[key] = initial_context[key]

        # Nettoyage : retirer les titres structurels du contenu
        contenu_nettoye = "\n".join(
            line.strip()
            for line in contenu_brut.splitlines()
            if line.strip() and not re.match(r"^(Livre|Titre|Chapitre|Section)\s+", line.strip(), re.IGNORECASE)
        )

        # Construction du préfixe contextuel (à intégrer dans le contenu final)
        context_lines = []
        if context["livre"]:
            context_lines.append(f"Livre : {context['livre']}")
        if context["titre"]:
            context_lines.append(f"Titre : {context['titre']}")
        if context["chapitre"]:
            context_lines.append(f"Chapitre : {context['chapitre']}")
        if context["section"]:
            context_lines.append(f"Section : {context['section']}")

        contenu_final = "\n".join(context_lines) + "\n\n" + contenu_nettoye

        # Ajout de l'article structuré
        articles.append({
            "titre_document": context["livre"],
            "titre_bloc": context["titre"],
            "division": context["chapitre"],
            "document_id": "",
            "permalien": article_id,
            "contenu": contenu_final
        })

    return articles

def build_metadata(block):
    return {
        "base": "Juridique",
        "source": "Code des assurances",
        "titre_document": block.get("titre_document", ""),
        "titre_bloc": block.get("titre_bloc", ""),
        "division": block.get("division", ""),
        "document_id": block.get("document_id", ""),
        "permalien": block.get("permalien", ""),
        "chunk_id": block.get("chunk_id", ""),
        "parent_chunk_id": block.get("parent_chunk_id", "")
    }

def split_with_overlap(blocks, max_tokens=800, overlap=0):
    chunks = []
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

            if para_token_len > max_tokens:
                for j in range(0, para_token_len, max_tokens):
                    sub_tokens = tokens_para[j:j + max_tokens]
                    sub_text = tokenizer.decode(sub_tokens)
                    chunk_id = str(uuid4())
                    block["chunk_id"] = chunk_id
                    chunks.append({
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "contenu": sub_text.strip(),
                        "metadata": build_metadata(block)
                    })
                    chunk_index += 1
                i += 1
                continue

            total_tokens = sum(len(t) for t in current_tokens)
            if total_tokens + para_token_len > max_tokens:
                chunk_text = "\n".join(current_chunk)
                chunk_id = str(uuid4())
                block["chunk_id"] = chunk_id
                chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "contenu": chunk_text.strip(),
                    "metadata": build_metadata(block)
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
            block["chunk_id"] = chunk_id
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "contenu": chunk_text.strip(),
                "metadata": build_metadata(block)
            })

    return chunks


def generate_small_chunks(big_chunks, max_tokens=100, min_tokens=30):
    small_chunks = []

    for chunk in big_chunks:
        content = chunk["contenu"]
        metadata = chunk["metadata"]

        # Préfixe contextuel (non compté dans les tokens)
        parts = []
        if metadata.get("titre_document"):
            parts.append(metadata["titre_document"].strip())
        if metadata.get("titre_bloc"):
            parts.append(metadata["titre_bloc"].strip())
        if metadata.get("division"):
            parts.append(metadata["division"].strip())
        prefix = "\n".join(parts).strip() + "\n\n" if parts else ""

        # Extraire la section depuis le contenu si présente dans les premières lignes
        lines = content.splitlines()
        for line in lines[:5]:  # On regarde uniquement les premières lignes
            if line.lower().startswith("section :"):
                parts.append(line.strip())
                break

        # Découpage par phrases uniquement sur le contenu
        sentences = tokenizer_fr.tokenize(content)
        current_chunk_sentences = []
        current_chunk_tokens = 0
        sub_index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            tokens = tokenizer.encode(sentence)
            token_len = len(tokens)

            # Si la phrase est trop longue
            if token_len > max_tokens:
                for i in range(0, token_len, max_tokens):
                    sub_tokens = tokens[i:i + max_tokens]
                    sub_text = tokenizer.decode(sub_tokens)
                    if len(sub_tokens) >= min_tokens:
                        full_text = prefix + sub_text.strip()
                        chunk_id = f"{metadata['chunk_id']}_{sub_index}"
                        enriched_metadata = metadata.copy()
                        enriched_metadata["chunk_id"] = chunk_id
                        enriched_metadata["parent_chunk_id"] = metadata["chunk_id"]
                        small_chunks.append({
                            "id": chunk_id,
                            "parent_chunk_id": metadata["chunk_id"],
                            "small_index": sub_index,
                            "contenu": full_text,
                            "metadata": enriched_metadata
                        })
                        sub_index += 1
                continue

            # Ajouter à la séquence actuelle
            if current_chunk_tokens + token_len <= max_tokens:
                current_chunk_sentences.append(sentence)
                current_chunk_tokens += token_len
            else:
                # Sauvegarder le chunk
                if current_chunk_tokens >= min_tokens:
                    chunk_text = " ".join(current_chunk_sentences)
                    full_text = prefix + chunk_text
                    chunk_id = f"{metadata['chunk_id']}__{sub_index}"
                    enriched_metadata = metadata.copy()
                    enriched_metadata["chunk_id"] = chunk_id
                    enriched_metadata["parent_chunk_id"] = metadata["chunk_id"]
                    small_chunks.append({
                        "id": chunk_id,
                        "parent_chunk_id": metadata["chunk_id"],
                        "small_index": sub_index,
                        "contenu": full_text.strip(),
                        "metadata": enriched_metadata
                    })
                    sub_index += 1

                # Réinitialiser
                current_chunk_sentences = [sentence]
                current_chunk_tokens = token_len

        # Dernier chunk s’il reste des phrases
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            if len(tokenizer.encode(chunk_text)) >= min_tokens:
                full_text = prefix + chunk_text
                chunk_id = f"{metadata['chunk_id']}__{sub_index}"
                enriched_metadata = metadata.copy()
                enriched_metadata["chunk_id"] = chunk_id
                enriched_metadata["parent_chunk_id"] = metadata["chunk_id"]
                small_chunks.append({
                    "id": chunk_id,
                    "parent_chunk_id": metadata["chunk_id"],
                    "small_index": sub_index,
                    "contenu": full_text.strip(),
                    "metadata": enriched_metadata
                })

    return small_chunks

def main():
    print("==== Chunking du Code des assurances ====\n")

    print("1. Extraction des articles depuis le PDF...")
    articles = extract_articles_from_pdf(PDF_PATH)
    print(f">> Articles extraits : {len(articles)}")

    print("2. Chunking en blocs...")
    big_chunks = split_with_overlap(articles, max_tokens=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f">> Chunks générés : {len(big_chunks)}")

    print("3. Sauvegarde des gros chunks...")
    OUTPUT_BIG_CHUNKS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_BIG_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(big_chunks, f, indent=2, ensure_ascii=False)
    print(f">> Fichier : {OUTPUT_BIG_CHUNKS.name}")

    print("4. Découpage en petits chunks...")
    small_chunks = generate_small_chunks(big_chunks)
    print(f">> Petits chunks générés : {len(small_chunks)}")

    print("5. Sauvegarde des petits chunks...")
    with open(OUTPUT_SMALL_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(small_chunks, f, indent=2, ensure_ascii=False)
    print(f">> Fichier : {OUTPUT_SMALL_CHUNKS.name}")

    print("\n✅ Chunking terminé.")

if __name__ == "__main__":
    main()
