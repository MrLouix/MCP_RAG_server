# MCP RAG Documents Server

Serveur MCP (Model Context Protocol) auto-hébergé pour l'agent Hermes Agent — base RAG locale avec **recherche sémantique**, **tagging hybride H1+H2**, **multi-workspace**, et **lazy-load/idle-unload** des modèles pour minimiser la RAM au repos.

## Architecture

```
Hermes Agent (stdio MCP)
        |
        v
  +-------------+
  |  server.py  |  FastMCP (15 outils)
  +-------------+
        |
        v
  +---------------------+
  |  IngestPipeline     |  extract + tag + chunk + embed + store
  |  WatchManager       |  watchdog + debounce + coalescence
  +---------------------+
        |
        v
  +-------------+    +-------------------+
  | ModelManager|    |  Storage (ChromaDB)
  | (lazy load) |    |  + SQLite path_index
  +-------------+    +-------------------+
        |
        +-- Embedder (sentence-transformers, 384 dims)
        +-- LLM tagger (llama.cpp, optionnel)
        +-- OCR (EasyOCR, optionnel)
```

## Fonctionnalités

- **Formats supportés** : PDF (texte + OCR fallback), images (OCR), TXT, Markdown, DOCX, CSV, XLSX
- **Tagging hybride** : H1 (règles heuristiques sur chemin/extension/nom de fichier) en quelques ms, H2 (LLM local) pour inférence sémantique (optionnel)
- **Multi-workspace** : collections ChromaDB séparées (`default`, `work`, etc.)
- **Mémoire** : lazy-load — modèles chargés à la première requête, déchargés après TTL configurable (idle unload). Au repos : **< 500 Mo**.
- **Debouncing** : ingestions par lot toutes les 10 secondes (évite les ingestions partielles lors de copies de répertoire)
- **15 outils MCP** : `ingest_directory`, `search_docs`, `get_document`, `list_documents`, `delete_document`, `clear_index`, `get_stats`, `watch_directory`, `get_tags`, `tag_document`, `reindex_all`, `diagnose`, `get_model_status`, `unload_models`, `preload_models`

## Installation rapide

```bash
git clone git@github.com:MrLouix/MCP_RAG_server.git
cd MCP_RAG_server
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

Copiez le fichier exemple :

```bash
cp config.example.yaml config.yaml
```

Éditez `config.yaml` :

```yaml
# Dossier de stockage de l'index ChromaDB + SQLite
index_path: "./rag_index"

# Modèle d'embedding (384 dims)
embedding_model: "paraphrase-multilingual-MiniLM-L12-v2"

# Lazy-load & idle-unload
lazy_load: true
idle_unload_after_s: 300

# Chunking
tokenizer: "cl100k_base"
chunk_size: 600
token_overlap: 120
max_chunks_per_doc: 500

# Tagging H2 (optionnel — nécessite un modèle GGUF local)
auto_tag: false
tagging_model: "/chemin/vers/mon_modele.gguf"

# OCR (optionnel)
ocr:
  enabled: false
  languages: ["fr", "en"]

# Sécurité
security:
  allow_dot_paths: false
  allowed_extensions: [".pdf", ".md", ".txt", ".docx", ".csv", ".xlsx", ".jpg", ".png"]

# Watch directory (optionnel)
watch:
  enabled: false
  directories: ["/home/user/documents"]
  recursive: true
  ignored_extensions: [".tmp", ".log", ".swp", ".bak"]
```

## Branchement sur Hermes Agent

1. Créez le fichier de configuration MCP d'Hermes (ex. `~/.hermes/mcp_rag.yaml`) :

```yaml
mcpServers:
  rag-documents:
    command: /chemin/vers/MCP_RAG_server/venv/bin/python
    args: [-m, mcp_rag.server, --config, /chemin/vers/MCP_RAG_server/config.yaml]
    env:
      PYTHONUNBUFFERED: "1"
      RAG_LOG_LEVEL: "INFO"
```

2. Redémarrez Hermes Agent :

```bash
# Selon votre setup Hermes
hermes-agent restart
# ou
systemctl --user restart hermes
```

3. Hermes découvrira automatiquement les 15 outils MCP exposés par le serveur et pourra :
   - **Ingestion** : "Indexe le dossier `/home/user/contrats` dans le workspace juridique"
   - **Recherche** : "Trouve les clauses de résiliation dans mes contrats"
   - **Statistiques** : "Combien de documents sont indexés ?"
   - **Tagging** : "Liste tous les tags du workspace comptabilité"
   - **Diagnostic** : "Vérifie la santé de l'index RAG"

## Utilisation CLI (sans Hermes)

### Ingestion batch

```bash
python -m mcp_rag.server --config config.yaml &
# Puis appeler les outils via MCP inspector ou le client Python
```

Ou directement via le pipeline Python :

```python
from mcp_rag.config import load_config
from mcp_rag.model_manager import ModelManager
from mcp_rag.storage import Storage
from mcp_rag.ingest import IngestPipeline

import asyncio

cfg = load_config()
pipeline = IngestPipeline(cfg, ModelManager(cfg), Storage(cfg.rag.index_path))
result = asyncio.run(pipeline.ingest_directory("/home/user/documents"))
print(result)
```

### Recherche directe

```python
from mcp_rag.embeddings import Embedder
from mcp_rag.storage import Storage

emb = Embedder(manager)
vectors = await emb.embed(["conditions de résiliation"])
hits = storage.search(vectors[0], top_k=5, workspace="default", embedding_model=await emb.get_model_name())
for h in hits:
    print(f"{h['metadata']['source_name']} | dist {h['distance']}")
    print(h['chunk_text'][:300])
```

## Paramètres importants

| Paramètre | Description | Par défaut |
|-----------|-------------|------------|
| `index_path` | Dossier de stockage ChromaDB + SQLite | `./rag_index` |
| `embedding_model` | Nom du modèle sentence-transformers | `paraphrase-multilingual-MiniLM-L12-v2` |
| `lazy_load` | Charge les modèles uniquement à la 1re requête | `true` |
| `idle_unload_after_s` | TTL avant déchargement (RAM ~ repose) | `300` |
| `chunk_size` | Tokens max par chunk | `600` |
| `token_overlap` | Chevauchement entre chunks | `120` |
| `auto_tag` | Active le tagging LLM H2 (nécessite modèle local) | `false` |
| `tagging_model` | Chemin vers le modèle GGUF (llama.cpp) | `""` |
| `ocr.enabled` | Active l'OCR EasyOCR pour images/PDF scannés | `false` |
| `security.allow_dot_paths` | Autorise l'indexation des chemins cachés | `false` |

## Inférence de tags H2 (optionnelle)

Le tagging H2 utilise un LLM local via `llama-cpp-python`. Il nécessite un modèle au format GGUF.

Pour activer :

```bash
# Télécharger un modèle GGUF (ex. Qwen2.5-Instruct 7B Q4)
mkdir -p models
curl -L -o models/qwen2.5-7b-instruct-q4_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_0.gguf
```

Puis éditez `config.yaml` :

```yaml
auto_tag: true
tagging_model: "./models/qwen2.5-7b-instruct-q4_0.gguf"
tagging_prompt: "default"
tagging_temperature: 0.1
```

⚠️ Le premier chargement du LLM peut prendre **1-3 minutes** en fonction du modèle.

## Tests

```bash
pytest tests/ -v
```

**17/17 tests unitaires passent** : hashing, chunker, model_manager (mocké), tagging heuristics, config loading.

### Validation E2E (manuelle)

Ingestion d'un fichier Markdown test → recherche sémantique "résiliation contrat" → 2 chunks pertinents retournés avec distance cosinus. Voir `docs/spec.md` §19 pour la procédure complète.

## Notes techniques

- **ChromaDB** : persistant, collection par workspace, validation du modèle d'embedding par collection (dimension fixe).
- **OCR** : fallback automatique pour les PDF contenant des pages sans texte sélectionable. EasyOCR charge ses modèles au premier appel (si `ocr: enabled: true`).
- **SQLite path_index** : index inversé `chemin absolu → doc_id` utilisé par le watcher et l'ingestion pour la déduplication.
- **RAM au repos** : GC thread toutes les 30 secondes, `idle_unload_after_s=300`, `gc.collect()` + `malloc_trim` après chaque unload.
- **Multi-workspace** : utilisé via le paramètre `workspace` sur tous les outils MCP (`default`, `juridique`, `compta`, etc.).

## Chiffrement

Le chiffrement n'est **pas implémenté** dans cette version (abandonné sur demande). Pour ajouter un chiffrement au repos, placez le dossier `index_path` sur un volume LUKS ou chiffrez individuellement les chunks via l'extension `encrypting-segment-transform` de ChromaDB.

## Docker

### Build

```bash
docker build -t mcp-rag-server:latest .
```

> **Image produite** : ~5.92 GB (multi-stage builder→production). Compilation torch + llama-cpp-python ≈ 5 min.
> **Dernier build validé** : 2026-05-06

### Run — Mode stdio (Hermès lance le conteneur)

```bash
docker run --rm \
  -e RAG_TRANSPORT=stdio \
  -v ./config.yaml:/app/config.yaml:ro \
  -v ./rag_index:/app/rag_index \
  mcp-rag-server:latest
```

### Run — Mode SSE (conteneur daemon, Hermès pointe sur HTTP)

```bash
docker run -d --name mcp-rag-server \
  -p 3000:3000 \
  -e RAG_TRANSPORT=sse \
  -e PORT=3000 \
  -e HOST=0.0.0.0 \
  -v ./config.yaml:/app/config.yaml:ro \
  -v ./rag_index:/app/rag_index \
  -v ./models:/app/models:ro \
  -v ~/documents:/app/documents:ro \
  mcp-rag-server:latest
```

### Docker Compose

```bash
# SSE mode
docker compose up -d

# Stopping
docker compose down
```

### Sécurité SSE

Le serveur FastMCP SSE expose `POST /messages` et `GET /sse`. Il est recommandé de placer un reverse-proxy (Caddy ou Nginx) devant, avec une restriction IP ou un token secret.

## Licence

MIT — Projet privé MrLouix.
