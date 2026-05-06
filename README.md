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

## Docker

C'est la méthode recommandée. Le serveur tourne dans un conteneur isolé avec SSE — Hermès s'y connecte via HTTP.

### Build

```bash
docker build -t mcp-rag-server:latest .
```

> **Image** : ~5.92 GB (multi-stage builder → production). Compilation torch + llama-cpp-python ≈ 5 min.

### 1. Préparer le dossier `config/`

Le conteneur a besoin d'un fichier `config.yaml`. Copiez-le depuis l'exemple :

```bash
mkdir -p config
cp config.example.yaml config/config.yaml
```

Éditez `config/config.yaml` selon vos besoins (voir [Configuration complète](#configuration-complète)). L'entrypoint créera automatiquement ce fichier au démarrage si absent, mais autant le préparer avant.

### 2. Préparer les dossiers de volumes

Le conteneur a besoin des dossiers suivants **créés sur le host** avant le démarrage :

```bash
mkdir -p config        # ← déjà fait ci-dessus
mkdir -p rag_index     # ← index ChromaDB (persisté)
mkdir -p documents     # ← vos fichiers à indexer
mkdir -p models        # ← modèles GGUF (optionnel, pour tagging H2)
```

### 3. Lancer avec Docker Compose

```bash
docker compose up -d
```

Vérifiez que le conteneur tourne :

```bash
docker compose ps
# → mcp-rag-server  Up (healthy)  0.0.0.0:3000->3000/tcp
```

Vérifiez les logs :

```bash
docker compose logs --tail 20
# → INFO server_initialized
# → INFO server_starting transport=sse
# → INFO sse_server_starting port=3000 host=0.0.0.0
```

### 4. Tester la connexion SSE

```bash
curl -N http://localhost:3000/sse &
PID=$!; sleep 2; kill $PID
# → event: endpoint
# → data: /messages/?session_id=...
```

Si vous voyez `event: endpoint` — le serveur est prêt.

### 5. Configurer Hermès pour utiliser le serveur

Ajoutez dans `~/.hermes/config.yaml` :

```yaml
mcp_servers:
  rag-documents:
    url: http://localhost:3000/sse
    timeout: 120
    connect_timeout: 30
```

Redémarrez Hermès. 15 nouveaux outils seront disponibles.

### Stop / Redémarrage

```bash
# Stop
docker compose down

# Redémarrage (après modif config)
docker compose down && docker compose up -d

# Rebuild + redémarrage (après modif code)
docker compose up -d --build

# Logs en continu
docker compose logs -f
```

### Structure des volumes sur le host

```
MCP_RAG_server/
├── config/          ← votre config.yaml personnalisé
├── rag_index/       ← index ChromaDB + SQLite (auto-créé au 1er lancement)
├── documents/       ← vos fichiers à indexer/watch
├── models/          ← modèles GGUF (optionnel, tagging H2)
├── docker-compose.yml
├── Dockerfile
├── config.example.yaml  ← template, ne pas modifier
└── README.md
```

---

## Watch Folder

Le **Watch Folder** surveille un répertoire du système de fichier et **indexe automatiquement** tout nouveau fichier ajouté, modifié, renommé ou supprimé — sans intervention manuelle.

### Comment ça marche

```
[documents/]  →  nouveau fichier PDF  →  watchdog détecte l'événement
                                            ↓ (debounce 2s)
                                       Ingestion automatique
                                            ↓
                                    extract → tag → chunk → embed → store
                                            ↓
                                    Document interrogeable via search_docs
```

1. **Détection** — watchdog surveille les événements `created`, `modified`, `deleted`, `moved`
2. **Debounce** — les événements sont regroupés toutes les 2 secondes (évite les ingestions partielles lors de copier-coller)
3. **Ingestion auto** — le fichier passe par le pipeline complet : extraction du texte, tagging H1/H2, chunking, embedding, stockage
4. **Suppression** — si un fichier est supprimé et `sync_deletions: true`, il est retiré de l'index

### Configuration dans `config/config.yaml`

```yaml
watcher:
  enabled: false              # Mettre true pour activer au démarrage du conteneur
  debounce_ms: 2000           # Regroupe les événements par tranches de 2s
  sync_deletions: true        # Supprime aussi de l'index quand un fichier est supprimé
  max_workers: 4              # Workers parallèles pour l'ingestion
  default_watch_paths:        # Liste des chemins à surveiller au démarrage
    - /app/documents          # ← le dossier monté dans Docker
    - /app/documents/contrats
  default_recursive: true     # Surveille aussi les sous-répertoires
```

Dans Docker Compose, le dossier est monté ici :

```yaml
volumes:
  - ./documents:/app/documents:ro   # Vos fichiers → accessible dans le conteneur
```

### Utilisation

**Mode automatique** (au démarrage du conteneur) :

1. Activez le watcher dans `config/config.yaml` : `enabled: true`
2. Ajoutez les chemins dans `default_watch_paths` (chemins **dans le conteneur**, ex: `/app/documents`)
3. Redémarrez : `docker compose down && docker compose up -d`

**Mode manuel** (via l'outil MCP) :

Appelez l'outil `watch_directory` depuis Hermès :

```
"Active la surveillance du dossier /app/documents dans le workspace par défaut"
```

Ou via le client MCP :

```json
{
  "name": "watch_directory",
  "arguments": {
    "dir_path": "/app/documents",
    "recursive": true,
    "enabled": true,
    "sync_deletions": true,
    "debounce_ms": 2000,
    "workspace": "default"
  }
}
```

### Exemple concret

```bash
# 1. Placez des fichiers dans le dossier
cp ~/mes_contrats/*.pdf /srv/MCP_RAG_server/documents/

# 2. Après ~2 secondes (debounce), le watcher les détecte et les indexe

# 3. Vérifiez que les fichiers sont dans l'index
# Via Hermès : "Combien de documents sont indexés ?"
# Ou via tool : get_stats → total_docs

# 4. Recherchez dedans
# Via Hermès : "Trouve les clauses de résiliation"
# Ou via tool : search_docs avec query="clauses résiliation"
```

### Notes importantes

- **Le dossier doit exister avant le démarrage du conteneur** — sinon le watcher logue un warning et skip ce chemin
- **Seuls les fichiers avec extensions supportées sont traités** : `.pdf`, `.md`, `.txt`, `.docx`, `.csv`, `.xlsx`, `.jpg`, `.png`
- **Fichiers temporaires ignorés** : les fichiers qui n'ont pas une extension supportée passent directement
- **Symlinks** — ne sont pas suivis par défaut
- **Taille** — pas de limite de taille de fichier, mais les fichiers très volumineux peuvent être lents à extraire + embed

## Installation locale (sans Docker)

```bash
git clone git@github.com:MrLouix/MCP_RAG_server.git
cd MCP_RAG_server
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Configuration complète

Copiez le fichier exemple :

```bash
cp config.example.yaml config.yaml
```

### Fichier `config.example.yaml`

```yaml
rag:
  index_path: "./rag_index"
  embedding:
    model: "paraphrase-multilingual-MiniLM-L12-v2"   # 384 dims
    backend: "sentence-transformers"
    fallback: "all-MiniLM-L6-v2"
  chunk_size: 600                 # tokens max par chunk
  chunk_overlap: 60               # chevauchement entre chunks
  max_chunks_per_doc: 1500        # limite de chunks par document
  ocr_enabled: true               # OCR pour PDF scannés / images
  ocr_languages: ["fra", "eng"]
  supported_extensions:
    - .pdf
    - .png
    - .jpg
    - .jpeg
    - .txt
    - .md
    - .markdown
    - .docx
    - .csv
    - .xlsx

memory:
  lazy_load: true                 # Charge les modèles à la 1re requête
  idle_ttl_embedder: 300          # TTL embedder avant unload (5 min)
  idle_ttl_llm: 120               # TTL LLM avant unload (2 min)
  idle_ttl_ocr: 180               # TTL OCR avant unload (3 min)
  gc_tick_seconds: 30             # Fréquence du GC thread
  aggressive_gc: true             # gc.collect() + malloc_trim après unload

tagging:
  auto_tag_enabled: true          # Active le tagging H1 (heuristique)
  model_path: ""                  # Chemin du modèle GGUF (H2, optionnel)
  n_ctx: 4096                     # Context window du LLM
  n_threads: 4                    # Threads CPU pour l'inférence LLM
  timeout_ms: 10000               # Timeout max par inférence
  use_cache: true                 # Cache SQLite pour éviter de retagger
  cache_path: ".rag_tag_cache.db"
  taxonomy:                       # Catégories pour le tagging H2
    domaine: ["financier", "juridique", "technique", "commercial", "rh", "administratif"]
    priorite: ["urgent", "normal", "faible"]
    confidentialite: ["public", "interne", "confidentiel"]
  temperature: 0.1
  seed: 42

watcher:
  enabled: false
  debounce_ms: 2000
  sync_deletions: true
  max_workers: 4
  default_watch_paths: []         # Ex: ["/app/documents"]
  default_recursive: true

security:
  ragrules_max_bytes: 102400      # Taille max .ragrules.yaml
  regex_timeout_ms: 200           # Timeout regex
  yaml_safe_load_only: true       # Chargement YAML sécurisé uniquement

logging:
  level: "INFO"
  format: "json"
  file: "./logs/mcp_rag.log"
  rotation: "50MB"
  retention_days: 14
```

| Paramètre | Description | Par défaut |
|-----------|-------------|------------|
| `index_path` | Dossier de stockage ChromaDB + SQLite | `./rag_index` |
| `embedding.model` | Modèle sentence-transformers (384 dims) | `paraphrase-multilingual-MiniLM-L12-v2` |
| `memory.lazy_load` | Charge les modèles uniquement à la 1re requête | `true` |
| `chunk_size` | Tokens max par chunk | `600` |
| `chunk_overlap` | Chevauchement entre chunks | `60` |
| `tagging.auto_tag_enabled` | Active tagging heuristique H1 | `true` |
| `tagging.model_path` | Chemin modèle GGUF (tagging H2 LLM, optionnel) | `""` |
| `watcher.enabled` | Active le watch folder au démarrage | `false` |
| `watcher.debounce_ms` | Délai de regroupement événements | `2000` |
| `ocr_enabled` | OCR EasyOCR pour PDF scannés / images | `true` |
| `security.allow_dot_paths` | Autorise l'indexation des chemins cachés | `false` |

## Tagging H2 avec LLM local (optionnel)

Le tagging H2 utilise un LLM local via `llama-cpp-python` pour inférer des tags sémantiques.

1. **Téléchargez un modèle GGUF** :

```bash
mkdir -p models
wget -P models \
  "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
```

2. **Activez dans `config.yaml`** :

```yaml
tagging:
  auto_tag_enabled: true
  model_path: "/app/models/qwen2.5-7b-instruct-q4_k_m.gguf"
```

3. **Montez le dossier `models/`** dans Docker Compose (déjà fait par défaut) :

```yaml
volumes:
  - ./models:/app/models:ro
```

⚠️ Le premier chargement du LLM peut prendre **1-3 minutes** selon le modèle. Le LLM est **déchargé automatiquement** après 120 s d'inactivité.

## Branchement sur Hermes Agent (mode stdio, sans Docker)

Ajoutez dans `~/.hermes/config.yaml` :

```yaml
mcp_servers:
  rag-documents:
    command: /chemin/vers/MCP_RAG_server/venv/bin/python
    args: [-m, mcp_rag.server, --config, /chemin/vers/MCP_RAG_server/config.yaml]
    env:
      PYTHONUNBUFFERED: "1"
      RAG_LOG_LEVEL: "INFO"
```

Redémarrez Hermes Agent.

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

## Sécurité SSE

Le serveur FastMCP SSE expose `POST /messages` et `GET /sse`. Il est recommandé de placer un reverse-proxy (Caddy ou Nginx) devant, avec une restriction IP ou un token secret.

## Licence

MIT — Projet privé MrLouix.
