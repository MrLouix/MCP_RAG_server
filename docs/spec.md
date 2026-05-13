# Spécification : Serveur MCP RAG Local — v4 (Ollama Backend + Auto-Tagging + Watch Folder)

## 1. Objectif

Créer un serveur MCP (Model Context Protocol) auto-hébergé qui expose une base de connaissances RAG à l'agent Hermes. Le système ingère des documents hétérogènes (PDF, images, texte brut, Markdown, DOCX, CSV/Excel), les **tague automatiquement** (heuristiques + LLM via Ollama), les indexe sémantiquement avec des embeddings via Ollama, et sert des recherches **filtrables par tags** via des outils MCP stdio. Un **watcher de dossier** synchronise l'index en temps réel avec le système de fichiers.

**Contraintes fortes :**
- **Aucune API cloud** (OpenAI, Anthropic, etc.) — tous les modèles LLM et d'embedding tournent sur une instance **Ollama externe** dont l'URL est configurable
- **Empreinte RAM minimale côté serveur MCP** : les modèles lourds tournent sur l'instance Ollama distante ; le serveur MCP ne charge que le client HTTP + ChromaDB
- Capable d'ingérer des centaines de fichiers en une seule commande sans bloquer le serveur MCP
- Métadonnées strictes et traçables (source, page, date, hash, tags système et sémantiques)
- L'index reflète toujours l'état du dossier surveillé (ajout/suppression automatiques)
- **Cohérence du modèle d'embedding** : figé par collection, migration explicite

---

## 2. Architecture Globale

```
┌─────────────────┐     ┌──────────────────────────────────┐     ┌─────────────────┐
│  Dossier Source │────▶│ Pipeline Ingest + Tagging Engine │────▶│  ChromaDB Local │
│  (PDF/IMG/TXT)  │     │  H1:Règles → H2:LLM (Ollama)    │     │  Persistent     │
│  ← sync auto ←  │◀────│  Watchdog (fs events queue)     │     │  (./rag_index/) │
│                 │     │  Ollama Client (httpx)          │     │                 │
│  ┌─.ragrules.yaml│     └────────────────┬─────────────────┘     └────────┬────────┘
│  └─tag_cache.db │                      │                               │
└─────────────────┘          ┌───────────▼──────────┐                    │ stdio
                             │  Serveur FastMCP     │◀───────────────────┤
                             │  (mcp_rag)           │                    │
                             │  + Ollama Client     │──── HTTP ──────┐   │
                             │  + Diagnose tools    │                │   │
                             └──────────┬───────────┘                │   │
                                        │                    ┌───────▼───────────┐
                                        ▼                    │  Ollama Server    │
                                ┌──────────────┐             │  (172.28.128.1)   │
                                │ Hermes Agent │             │  :11434           │
                                │ (Client MCP) │             │  LLM + Embed +   │
                                └──────────────┘             │  Vision models   │
                                                             └───────────────────┘
```

**Flux nominal :**
1. L'utilisateur place ses docs dans un dossier source → le **watcher** les détecte
2. Le pipeline appelle l'instance **Ollama externe** via HTTP pour embedding, tagging H2, et OCR/vision
3. Le pipeline tourne : extraction → H1 → H2 (Ollama) → chunking → embedding (Ollama) → stockage
4. L'agent interroge via `search_docs` → l'embedding de la requête est calculé via Ollama → résultats filtrés
5. Les suppressions filesystem sont répercutées automatiquement si `sync_deletions=true`

**Changement majeur vs v3 :** les modèles (LLM, embedding, vision/OCR) ne sont **plus chargés en RAM** par le serveur MCP. Ils tournent sur une instance Ollama externe. Le serveur MCP est un **client HTTP léger** (~200–400 MB de RAM).

---

## 3. Pipeline d'Ingestion Batch

### 3.1 Extraction par format

| Format   | Bibliothèque             | Stratégie                                                                 |
|----------|--------------------------|---------------------------------------------------------------------------|
| PDF texte| `pdfplumber` / `PyMuPDF` | Extraction texte page par page, conservation structure                    |
| PDF scanné | `PyMuPDF` + **Ollama vision** | Détection pages sans texte → envoi image de la page au modèle vision Ollama → reconstruction texte |
| Images   | `Pillow` + **Ollama vision** | Envoi de l'image au modèle vision Ollama pour extraction de texte (prompt OCR dédié) |
| Markdown | `pathlib` / `markdown`   | Parsing natif, conservation des liens internes et blocs de code           |
| TXT      | `pathlib`                | Lecture directe, détection encodage (UTF-8 fallback Latin-1)              |
| DOCX     | `python-docx`            | Extraction texte des paragraphes, tableaux, styles                        |
| CSV/Excel| `pandas` / `openpyxl`    | Lecture structurée, concaténation des colonnes pertinentes                |

**Note OCR/Vision :** l'OCR via EasyOCR est remplacé par un modèle **vision multimodal** servi par Ollama (ex. `llava`, `minicpm-v`, `moondream`). L'image est envoyée en base64 dans l'appel API `/api/chat` avec un prompt demandant l'extraction du texte. Cela élimine la dépendance PyTorch/CUDA côté serveur MCP.

### 3.2 Nettoyage & Normalisation
- Suppression en-têtes/pieds de page répétitifs (détection par fréquence)
- Normalisation unicode (`unicodedata.normalize`)
- Suppression caractères non-imprimables, espaces multiples
- Conservation des sauts de paragraphes significatifs

### 3.3 Stratégie de Chunking
- **Splitter** : `RecursiveCharacterTextSplitter` (LangChain)
- **Taille cible** : ~600 tokens / chunk
- **Overlap** : 10 % (~60 tokens) pour préserver le contexte transversal
- **Délimitation** : paragraphes > sections > phrases > mots
- **Métadata par chunk** : format enrichi par les tags (voir §5)
- **Limite de sûreté** : `max_chunks_per_doc = 1500` — au-delà, warning dans les logs et dans le retour d'ingestion (PDF de plusieurs milliers de pages)

---

## 4. Tagging Automatique (Hybride H1 + H2)

Le tagging est **hybride** et produit deux familles de tags : les tags *système* (déterministes) et les tags *sémantiques* (inférés par un LLM via Ollama).

### 4.1 Couche H1 — Tags Système (Heuristiques)

Évaluation immédiate (< 10 ms), sans LLM :

- **Chemin et nom de fichier** : regex configurables (ex. `/clients/(?P<client>[^/]+)/` → `client:acme`)
- **Extension / MIME** : `format:pdf`, `format:md`, `format:docx`
- **Règles utilisateur** : fichier `.ragrules.yaml` optionnel à la racine du dossier ingéré
- **Métadonnées natives** : dates de création, taille du fichier
- **Répertoires parent** : segments de chemin → tags automatiques (`year:2026`, `projet:alpha`)

Exemple `.ragrules.yaml` :
```yaml
rules:
  - pattern: "facture*"
    tags: ["type:facture"]
  - pattern: "**/juridique/**"
    tags: ["domaine:juridique", "confidentialite:interne"]
  - pattern: "*.md"
    tags: ["type:note"]
```

### 4.2 Couche H2 — Tags Sémantiques (LLM via Ollama)

- **Backend** : instance Ollama externe, accessible via HTTP (`/api/chat` avec `format: "json"`)
- **URL configurable** : `ollama.base_url` (défaut `http://172.28.128.1:11434`)
- **Modèle de tagging** : configurable via `ollama.tag_model` (défaut `qwen2.5:3b`)
- **Modèles recommandés** : `qwen2.5:3b`, `phi4-mini`, `gemma2:2b` — modèles légers avec bon support JSON
- **Mode de sortie** : **JSON Mode** natif Ollama (`format: "json"` dans l'appel API) — pas de sortie libre
- **Concurrence** : les requêtes sont sérialisées côté client (un seul appel Ollama tagging à la fois) pour éviter la contention GPU/CPU sur l'instance Ollama. Un `asyncio.Semaphore(1)` contrôle la concurrence.

**Taxonomie par défaut** (surchargeable via `ingest_directory.auto_tag.taxonomy`) :
```json
{
  "domaine": ["financier", "juridique", "technique", "commercial", "rh", "administratif"],
  "priorite": ["urgent", "normal", "faible"],
  "langue": "ISO 639-1",
  "entites": ["array", "string"],
  "confidentialite": ["public", "interne", "confidentiel"]
}
```

**Inférence granulaire** : le tagging H2 se fait **une seule fois par document** (avant le chunking). L'input LLM contient :
- Nom du fichier et chemin relatif
- Tags H1 déjà connus
- Les ~1500 premiers tokens du document (ou les 3 premiers chunks concaténés)

**Prompt type** :
```text
Tu es un classifieur de documents. Analyse le document ci-dessous et réponds UNIQUEMENT par un objet JSON valide respectant ce schéma:
{
  "domaine": "un parmi [financier, juridique, technique, commercial, rh, administratif]",
  "priorite": "un parmi [urgent, normal, faible]",
  "langue": "code ISO 639-1",
  "entites": ["3 à 5 mots-clés propres au contenu"],
  "confidentialite": "un parmi [public, interne, confidentiel]"
}

Règles:
- Ne produis aucun texte hors du JSON.
- Si l'information est absente, utilise null.

Document:
--- Début ---
{extrait_1500_tokens}
--- Fin ---
```

**Appel API Ollama** :
```json
POST {ollama.base_url}/api/chat
{
  "model": "qwen2.5:3b",
  "messages": [{"role": "user", "content": "<prompt>"}],
  "format": "json",
  "stream": false,
  "options": {
    "temperature": 0.1,
    "seed": 42,
    "num_predict": 256
  }
}
```

### 4.3 Cache de Tags

Base SQLite locale `.rag_tag_cache.db` (préférée à JSONL pour la concurrence lecture/écriture) :
- Mapping `content_hash (SHA-256) → {tags_json, model_version, inferred_at}`
- Si un fichier est ré-ingéré et que son hash n'a pas changé, les tags H2 sont recyclés instantanément
- Invalidation automatique si le modèle de tagging Ollama change (`model_version` = `"{tag_model}:{taxonomy_hash}"`)

### 4.4 Fallback et Robustesse

- Si l'instance Ollama est injoignable ou timeout (défaut **30 s**, configurable) → ingestion se poursuit avec uniquement les tags H1
- Le timeout est strict : pas de blocage du serveur MCP stdio
- Un warning est logué avec le chemin du fichier ayant échoué au tagging H2 (`llm_status: "timeout"` ou `"unreachable"` dans les métadonnées)
- **Healthcheck Ollama** : au démarrage et dans `diagnose`, un `GET {ollama.base_url}/api/tags` vérifie la connectivité et la présence des modèles requis

---

## 5. Modèle de Données : JSON Chunk Enrichi

Chaque chunk stocké dans le vector store respecte le format suivant :

```json
{
  "schema_version": "1.0",
  "doc_id": "sha256_prefix_8chars",
  "source_path": "/abs/path/to/file.pdf",
  "source_name": "file.pdf",
  "page_or_section": 3,
  "chunk_index": 12,
  "total_chunks": 45,
  "ingested_at": "2026-05-06T14:30:00Z",
  "modified_at": "2026-04-20T09:00:00Z",
  "file_type": "pdf",
  "content_hash": "sha256_full",
  "embedding_model": "nomic-embed-text",
  "orphaned": false,
  "tags": {
    "system": ["client:acme", "year:2026", "type:facture", "format:pdf"],
    "semantic": ["domaine:financier", "priorite:normal", "langue:fr", "entite:tv_a_credit"],
    "model": "qwen2.5:3b",
    "inferred_at": "2026-05-06T14:30:01Z",
    "llm_status": "ok"
  }
}
```

**Notes :**
- `schema_version` permet de futures migrations sans casser l'index
- `embedding_model` est répliqué à chaque chunk : on peut détecter une incohérence si l'index est partagé
- `llm_status ∈ {"ok", "timeout", "error", "unreachable", "disabled"}`
- Les **tags sont calculés au niveau documentaire** mais **répliqués dans chaque chunk**, permettant aux filtres `where` du vector store de fonctionner même sur un fragment isolé

---

## 6. Pipeline d'Ingestion Complet (Flux)

```
Fichier détecté (watcher ou ingest_directory)
        │
        ▼
┌──────────────────┐     ┌──────────────┐     ┌──────────────┐
│ Hash SHA-256     │────▶│ Cache Tags ? │────▶│ Extraction   │
│ & vérif existance│     └──────────────┘     │ texte brut   │
└──────────────────┘          │               └──────────────┘
    ▲                  [Hit & inchangé]               │
    │                   │ oui │    │non│              ▼
    │                   ▼          ▼          ┌──────────────┐
    │             ┌─────────┐   ┌──────────────┐│ Tagging H1   │
    │             │ Skip ou │   │ Cache raté : ││ (règles <10ms)│
    │             │ Réindex │   │ Extraction   │└──────────────┘
    │             └─────────┘   │ texte + H2   │      │
    │                           │(Ollama HTTP) │      ▼
    └───────────────────────────┴──────────────┘┌──────────────┐
                                                │ Merge H1+H2  │
                                                │ + cache tags │
                                                └──────────────┘
                                                        │
                                                        ▼
                                               ┌──────────────┐
                                               │ Chunking     │
                                               │ Recursive    │
                                               │ Character    │
                                               └──────────────┘
                                                        │
                                                        ▼
                                               ┌──────────────────┐
                                               │ Embedding        │
                                               │ (Ollama /api/    │
                                               │  embed, batch 32)│
                                               └──────────────────┘
                                                        │
                                                        ▼
                                               ┌──────────────┐
                                               │ Write Chroma │
                                               │ (bulk insert)│
                                               └──────────────┘
```

**Sérialisation par `doc_id` :** un verrou applicatif (lock en mémoire, keyed par `doc_id`) empêche l'ingestion concurrente d'un même fichier (cas : watcher `modified` qui arrive pendant une ingestion batch). Les évts en double sur le même `doc_id` sont coalescés.

---

## 7. Client Ollama et Gestion des Modèles

### 7.1 Principe

Le serveur MCP ne charge **aucun modèle en RAM locale**. Tous les appels d'inférence (embedding, tagging, vision/OCR) passent par une instance Ollama externe via HTTP. Le serveur MCP est un client léger (~200–400 MB de RAM totale).

### 7.2 Modèles Ollama utilisés

| Rôle | Endpoint Ollama | Modèle par défaut | Config key | Note |
|------|----------------|-------------------|------------|------|
| **Embedding** | `POST /api/embed` | `nomic-embed-text` | `ollama.embed_model` | 768 dims, contexte 8192 tokens |
| **Tagging H2** | `POST /api/chat` (JSON mode) | `qwen2.5:3b` | `ollama.tag_model` | Sortie JSON contrainte |
| **Vision/OCR** | `POST /api/chat` (multimodal) | `minicpm-v` | `ollama.vision_model` | Images en base64, extraction texte |

### 7.3 Client HTTP (`OllamaClient`)

```python
class OllamaClient:
    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        """Client HTTP async (httpx) vers l'instance Ollama."""

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """POST /api/embed — retourne les vecteurs d'embedding."""

    async def chat(self, model: str, messages: list[dict], format: str = "json", options: dict | None = None) -> str:
        """POST /api/chat — retourne le contenu de la réponse."""

    async def chat_vision(self, model: str, prompt: str, images: list[str], options: dict | None = None) -> str:
        """POST /api/chat avec images base64 — pour OCR/vision."""

    async def list_models(self) -> list[str]:
        """GET /api/tags — retourne les modèles disponibles."""

    async def healthcheck(self) -> dict:
        """Vérifie la connectivité et la présence des modèles requis."""

    async def pull_model(self, model: str) -> dict:
        """POST /api/pull — télécharge un modèle si absent (optionnel)."""
```

**Paramètres HTTP :**
- Client `httpx.AsyncClient` avec `timeout` configurable (défaut 30 s pour les requêtes, 120 s pour les embeddings de gros batch)
- Retry avec backoff exponentiel (3 tentatives) en cas d'erreur réseau
- Connection pooling (keep-alive par défaut via httpx)

### 7.4 Gestion mémoire simplifiée

Puisque les modèles tournent sur l'instance Ollama externe :
- **Pas de `ModelManager` lourd** : remplacé par un `OllamaClient` stateless
- **Pas de thread GC** : la gestion mémoire des modèles est déléguée à Ollama (qui a ses propres TTL via `OLLAMA_KEEP_ALIVE`)
- **`preload_models`** : envoie un appel dummy à Ollama pour forcer le chargement du modèle en VRAM/RAM sur le serveur Ollama
- **`unload_models`** : envoie `POST /api/chat` avec `keep_alive: 0` pour demander le déchargement côté Ollama
- **`get_model_status`** : interroge `GET /api/ps` (Ollama running models) pour connaître les modèles actuellement chargés côté Ollama

### 7.5 Outils MCP de gestion modèles

- `get_model_status` : retourne les modèles chargés sur l'instance Ollama (`/api/ps`) + connectivité
- `unload_models` : envoie des requêtes `keep_alive: 0` à Ollama pour libérer la VRAM/RAM distante
- `preload_models` : envoie un appel warm-up à Ollama pour pré-charger les modèles

### 7.6 Configuration

```yaml
ollama:
  base_url: "http://172.28.128.1:11434"
  embed_model: "nomic-embed-text"
  tag_model: "qwen2.5:3b"
  vision_model: "minicpm-v"
  timeout_s: 30              # timeout par requête
  embed_timeout_s: 120       # timeout pour gros batch d'embeddings
  max_retries: 3
  auto_pull: false            # si true, pull automatiquement les modèles manquants
```

### 7.7 Compromis et trade-offs

- **Pros** : empreinte RAM serveur MCP minimale (~200–400 MB), pas de PyTorch/CUDA côté serveur, modèles gérés centralement par Ollama (partageables entre services), GPU offloading natif Ollama
- **Cons** : dépendance réseau vers l'instance Ollama (latence ~5–50 ms par requête), l'instance Ollama doit être démarrée et avoir les modèles pull
- **Mitigation** : healthcheck au démarrage, fallback H1-only si Ollama injoignable, `auto_pull` optionnel, retry avec backoff

---

## 8. Synchronisation Temps Réel (Watch Folder)

### 8.1 Filesystem Watcher

- **Librairie** : `watchdog` (cross-platform, abstraction de `inotify` / `ReadDirectoryChangesW` / `FSEvents`)
- **Événements surveillés** : `created`, `modified`, `moved`, `deleted`
- **Debouncing** : fenêtre de `debounce_ms` (défaut 2000 ms) pour éviter d'ingérer un fichier en cours d'écriture
- **Queue asynchrone** : les événements sont mis en file et traités par un pool de workers (**extraction/embedding parallélisables**, tagging H2 strictement sérialisé via le sémaphore Ollama)

### 8.2 Cycle de vie des fichiers

| Événement filesystem | Réaction |
|----------------------|----------|
| **Création** (`created`) | Ingestion complète pipeline (H1 + H2 → chunk → embed → store) |
| **Modification** (`modified`) | Hash SHA-256 changé → suppression ancienne version + ré-ingestion complète |
| **Renommage** (`moved`) | Mise à jour `source_path` et `source_name` dans les métadonnées ; hash inchangé → skip ré-embedding |
| **Suppression** (`deleted`) | Si `sync_deletions=true` → suppression du `doc_id` dans ChromaDB via index inversé `path → doc_id`<br>Si `sync_deletions=false` → document marqué `orphaned: true` |

### 8.3 Index inversé

Une table SQLite locale (mapping `absolute_path → doc_id`) maintient la correspondance entre le système de fichiers et les documents indexés :
- Permet la suppression ciblée lors d'un événement `deleted`
- Permet la détection de déplacements (`moved`)
- Persisté dans `./rag_index/path_index.db`

### 8.4 Verrous et race conditions

- **Verrou par `doc_id`** : `asyncio.Lock` par doc pour sérialiser les opérations concurrentes
- **Coalescence d'événements** : plusieurs `modified` rapprochés sur le même fichier → traité une seule fois après le debounce
- **Batch vs watcher** : pendant un `ingest_directory`, les événements watcher sur les fichiers du même dossier sont mis en attente jusqu'à la fin du batch

---

## 9. Embeddings & Base Vectorielle

### 9.1 Modèle d'Embedding (via Ollama)

- **Backend unique** : Ollama (`POST /api/embed`)
- **Modèle par défaut** : `nomic-embed-text` (768 dimensions, contexte 8192 tokens)
- **Modèles alternatifs** : `mxbai-embed-large` (1024 dims), `all-minilm` (384 dims), `snowflake-arctic-embed` (1024 dims)
- **Configurable** via `ollama.embed_model` dans `config.yaml`
- **Batch** : l'API Ollama `/api/embed` accepte un tableau d'inputs ; le serveur MCP envoie des batches de 32 textes max

### 9.2 Stockage Vectoriel

- **Moteur** : `chromadb` (PersistentClient)
- **Répertoire** : `./rag_index/` à la racine du projet
- **Collection** : `documents` (ou collections nommées pour multi-workspace, voir §9.4)
- **Fonction de distance** : Cosine (défaut ChromaDB)
- **Capacité cible** : ~50 000 chunks (500 docs × ~100 chunks/doc), testé jusqu'à 200 000

### 9.3 Cohérence du modèle d'embedding (critique)

ChromaDB ne permet pas de mélanger des vecteurs de dimensions différentes dans une même collection.

**Règle :**
- Le modèle d'embedding est **figé** à la création de la collection (stocké dans `collection.metadata`)
- Tout `ingest_directory` ultérieur **vérifie** que le modèle configuré correspond
- En cas de divergence → erreur explicite avec suggestion : `reindex_all` ou `clear_index`
- L'outil `get_stats` retourne toujours le modèle actif de la collection

**Migration :**
- Outil `reindex_all(new_embedding_model, confirm=true)` : vide la collection, reparcourt tous les fichiers listés dans l'index inversé, réembedded avec le nouveau modèle via Ollama
- Les tags H2 sont **recyclés depuis le cache SQLite** (pas de nouvelle inférence LLM)

### 9.4 Multi-workspace (optionnel)

- Paramètre `workspace: str` dans `ingest_directory`, `search_docs`, `list_documents`, etc.
- Défaut : `"default"` — collection `documents`
- Permet d'isoler `work`, `perso`, `client_X` dans des collections ChromaDB séparées

---

## 10. Interface MCP (Outils Exposés)

Tous les outils retournent du JSON strict. Paramètres nouveaux indiqués en **gras**.

| Outil | Description | Paramètres principaux | Retour |
|-------|-------------|----------------------|--------|
| `ingest_directory` | Indexe un dossier avec tagging auto | `dir_path`, `recursive`, `auto_tag`, `tag_rules_path`, `workspace` | `{status, ingested, skipped, errors, tagging_stats}` |
| `search_docs` | Recherche sémantique filtrable | `query`, `tags`, `tags_mode`, `filters`, `top_k`, `workspace` | `[{text, metadata, distance}]` |
| `get_document` | Récupère chunks d'un doc | `doc_id`, `workspace` | `{doc_id, chunks: [...]}` |
| `list_documents` | Liste paginée filtrable | `limit`, `offset`, `tags`, `tags_mode`, `include_orphaned`, `workspace` | `{documents, total_count, returned}` |
| `delete_document` | Supprime un doc | `doc_id`, `workspace` | `{status, removed_chunks}` |
| `clear_index` | Vide l'index | `confirm`, `workspace` | `{status}` |
| `get_stats` | État de l'index | `workspace` (optionnel) | `{total_docs, total_chunks, embedding_model, tag_model, active_watchers, model_status}` |
| `watch_directory` | Active/désactive la sync d'un dossier | `dir_path`, `recursive`, `enabled`, `sync_deletions`, `debounce_ms`, `workspace` | `{status, watcher_id, watched_paths}` |
| `get_tags` | Taxonomie complète avec comptage | `query` (filtre), `workspace` | `{tags: [{tag, count, origin}]}` |
| `tag_document` | Retaguer un fichier sans ré-ingestion | `doc_id` ou `file_path`, `force_retag`, `custom_taxonomy`, `workspace` | `{status, tags, duration_ms, cache_hit}` |
| `reindex_all` | Réindexe tous les docs avec nouveau modèle | `new_embedding_model`, `confirm`, `workspace` | `{status, reindexed, duration_ms}` |
| `diagnose` | Healthcheck complet | `workspace` (optionnel) | `{ollama, disk, chroma, watchers, orphans, warnings}` |
| `get_model_status` | État des modèles sur Ollama | *(aucun)* | `{ollama_url, models_running, embed_model, tag_model, vision_model}` |
| `unload_models` | Demande le déchargement côté Ollama | `models: list[str]` (défaut: tous) | `{unloaded, status}` |
| `preload_models` | Précharge les modèles côté Ollama | `models: list[str]` | `{loaded, status}` |

**Harmonisation des paramètres :** tous les outils qui filtrent par tags utilisent `tags: list[str]` + `tags_mode: "all" | "any" | "exclude"`. Plus de `scope_mode` ni `match_mode` distincts.

**Conventions MCP :**
- Docstrings obligatoires sur chaque `@_mcp.tool()`
- Typage strict des paramètres
- Retour systématique en `dict` JSON-sérialisable
- Gestion d'erreurs : jamais de stacktrace brute, messages explicites

---

## 11. Détail des Outils MCP

### 11.1 `ingest_directory`

```json
{
  "dir_path": "/chemin/vers/documents",
  "recursive": true,
  "workspace": "default",
  "chunk_size": 600,
  "chunk_overlap": 60,
  "auto_tag": {
    "enabled": true,
    "taxonomy": {
      "domaine": ["financier", "juridique", "technique", "commercial", "rh"],
      "priorite": ["urgent", "normal", "faible"],
      "confidentialite": ["public", "interne", "confidentiel"]
    },
    "timeout_ms": 30000,
    "use_cache": true
  },
  "tag_rules_path": "/chemin/.ragrules.yaml",
  "unload_after": false
}
```

`unload_after: true` → envoie `keep_alive: 0` à Ollama pour libérer les modèles après le batch.

**Retour :**
```json
{
  "status": "success",
  "workspace": "default",
  "ingested": 142,
  "skipped": 8,
  "errors": 2,
  "tagging_stats": {
    "cache_hits": 50,
    "llm_inferences": 90,
    "llm_failures": 2,
    "avg_inference_ms": 520
  },
  "duration_ms": 12450
}
```

### 11.2 `search_docs`

```json
{
  "query": "conditions de résiliation du contrat",
  "tags": ["client:acme", "domaine:juridique"],
  "tags_mode": "any",
  "filters": {
    "file_type": "pdf",
    "ingested_at": { "$gte": "2026-01-01" }
  },
  "top_k": 5,
  "workspace": "default"
}
```

**`tags_mode`** :
- `"all"` : le document doit posséder **tous** les tags listés
- `"any"` (défaut) : le document doit posséder **au moins un** des tags
- `"exclude"` : aucun des tags listés ne doit être présent
- **Absent ou `tags = []`** : recherche globale sans filtre tag

### 11.3 `list_documents`

```json
{
  "limit": 20,
  "offset": 0,
  "tags": ["client:acme", "type:facture"],
  "tags_mode": "all",
  "include_orphaned": false,
  "workspace": "default"
}
```

### 11.4 `watch_directory`

```json
{
  "dir_path": "/chemin/vers/documents",
  "recursive": true,
  "enabled": true,
  "sync_deletions": true,
  "debounce_ms": 2000,
  "workspace": "default"
}
```

**Retour :**
```json
{
  "status": "watching",
  "watcher_id": "wd_01",
  "watched_paths": ["/chemin/vers/documents"],
  "recursive": true,
  "sync_deletions": true
}
```

### 11.5 `get_tags`

```json
{ "query": null, "workspace": "default" }
```

**Retour :**
```json
{
  "tags": [
    { "tag": "client:acme", "count": 24, "origin": "system" },
    { "tag": "domaine:financier", "count": 31, "origin": "semantic" }
  ],
  "total_tags": 2
}
```

### 11.6 `tag_document`

Retaguer un fichier spécifique sans le ré-ingérer entièrement.

```json
{
  "doc_id": "a1b2c3d4",
  "force_retag": true,
  "custom_taxonomy": {
    "sentiment": ["positif", "negatif", "neutre"]
  },
  "workspace": "default"
}
```

### 11.7 `reindex_all`

```json
{
  "new_embedding_model": "mxbai-embed-large",
  "confirm": true,
  "workspace": "default"
}
```

**Retour :**
```json
{
  "status": "success",
  "previous_model": "nomic-embed-text",
  "new_model": "mxbai-embed-large",
  "reindexed": 142,
  "tag_cache_hits": 142,
  "llm_inferences": 0,
  "duration_ms": 184300
}
```

### 11.8 `diagnose`

Healthcheck opérationnel. Ne charge **aucun modèle**.

**Retour :**
```json
{
  "status": "ok",
  "ollama": {
    "reachable": true,
    "base_url": "http://172.28.128.1:11434",
    "models_available": ["nomic-embed-text", "qwen2.5:3b", "minicpm-v"],
    "models_running": ["nomic-embed-text"]
  },
  "chroma": { "reachable": true, "collections": ["default", "perso"] },
  "disk": { "index_size_mb": 842, "free_gb": 128, "warning": null },
  "watchers": [{ "id": "wd_01", "path": "/data/docs", "active": true, "queue_size": 0 }],
  "orphans": { "count": 3, "examples": ["/data/docs/obsolete.pdf"] },
  "warnings": ["3 documents orphans détectés"]
}
```

### 11.9 `get_model_status` / `unload_models` / `preload_models`

```json
// get_model_status : aucune entrée
{
  "ollama_url": "http://172.28.128.1:11434",
  "ollama_reachable": true,
  "models_running": [
    { "name": "nomic-embed-text", "size_mb": 274 },
    { "name": "qwen2.5:3b", "size_mb": 1900 }
  ],
  "embed_model": "nomic-embed-text",
  "tag_model": "qwen2.5:3b",
  "vision_model": "minicpm-v"
}

// unload_models
{ "models": ["qwen2.5:3b", "minicpm-v"] }
// → { "unloaded": ["qwen2.5:3b", "minicpm-v"], "status": "ok" }

// preload_models
{ "models": ["nomic-embed-text", "qwen2.5:3b"] }
// → { "loaded": ["nomic-embed-text", "qwen2.5:3b"], "status": "ok" }
```

---

## 12. Configuration & Paramétrage

Fichier `config.yaml` :

```yaml
ollama:
  base_url: "http://172.28.128.1:11434"
  embed_model: "nomic-embed-text"
  tag_model: "qwen2.5:3b"
  vision_model: "minicpm-v"
  timeout_s: 30
  embed_timeout_s: 120
  max_retries: 3
  auto_pull: false

rag:
  index_path: "./rag_index"
  chunk_size: 600
  chunk_overlap: 60
  max_chunks_per_doc: 1500
  ocr_enabled: true
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

tagging:
  auto_tag_enabled: true
  timeout_ms: 30000
  use_cache: true
  cache_path: ".rag_tag_cache.db"
  taxonomy:
    domaine: ["financier", "juridique", "technique", "commercial", "rh", "administratif"]
    priorite: ["urgent", "normal", "faible"]
    langue: "ISO 639-1"
    entites: ["array", "string"]
    confidentialite: ["public", "interne", "confidentiel"]
  temperature: 0.1
  seed: 42

watcher:
  enabled: false
  debounce_ms: 2000
  sync_deletions: true
  max_workers: 4
  default_watch_paths: []
  default_recursive: true

security:
  ragrules_max_bytes: 102400        # 100 KB
  regex_timeout_ms: 200
  yaml_safe_load_only: true

logging:
  level: "INFO"
  format: "json"
  file: "./logs/mcp_rag.log"
  rotation: "50MB"
  retention_days: 14
```

Chargement via `pydantic-settings` ou simple `yaml.safe_load`.

**Variables d'environnement** : toutes les clés sont surchargeables via `MCP_RAG__OLLAMA__BASE_URL`, `MCP_RAG__OLLAMA__TAG_MODEL`, etc.

---

## 13. Contraintes & Performance

| Aspect | Cible | Note |
|--------|-------|------|
| **RAM serveur MCP (idle)** | **< 400 MB** | Python + ChromaDB client + watchers uniquement (pas de modèles locaux) |
| **RAM serveur MCP (actif)** | **< 600 MB** | Buffers de texte/embeddings en transit + ChromaDB |
| **Première requête (Ollama cold)** | +5 à 30 s | Chargement du modèle côté Ollama (dépend du modèle et du hardware distant) |
| **Ingestion batch** | 300 docs / 10 – 30 min | Parallélisation extraction, embedding via Ollama batch, tagging H2 sérialisé |
| **Recherche (chaud)** | < 500 ms (top_k=5) | Embedding de la query via Ollama (~50 ms) + recherche ChromaDB |
| **Tagging H2** | 200 – 2000 ms / doc | Dépend du modèle et du hardware Ollama |
| **Watch folder latence** | < 3 s | Debounce 2s + traitement |
| **Portabilité** | Linux / macOS / WSL | Dépendances Python uniquement (pas de PyTorch, pas de CUDA côté serveur) |
| **Reproductibilité** | Hash SHA-256, seed LLM fixe | Détection doublons, tagging quasi-déterministe |

---

## 14. Structure du Projet

```
mcp-rag-server/
├── docs/
│   ├── spec.md                     # Ce fichier
│   └── plans/                      # Plans d'implémentation détaillés
├── mcp_rag/
│   ├── __init__.py
│   ├── server.py                   # FastMCP + outils (@_mcp.tool)
│   ├── ollama_client.py            # Client HTTP async vers Ollama (§7.3)
│   ├── ingest.py                   # Pipeline extraction/chunking/embedding
│   ├── extractors.py               # Lecteurs par format (PDF, IMG, TXT, MD, DOCX, CSV)
│   ├── chunker.py                  # Stratégie de segmentation
│   ├── embeddings.py               # Wrapper embedding via Ollama /api/embed
│   ├── storage.py                  # Abstraction ChromaDB + index inversé
│   ├── config.py                   # Pydantic/YAML configuration
│   ├── tagging/
│   │   ├── __init__.py
│   │   ├── engine.py               # Orchestrateur H1 + H2
│   │   ├── heuristics.py           # H1 : règles, regex, .ragrules.yaml
│   │   ├── llm_tagger.py           # H2 : Ollama /api/chat JSON mode, cache SQLite
│   │   └── taxonomy.py             # Définition et validation
│   ├── watcher/
│   │   ├── __init__.py
│   │   └── fs_watcher.py           # watchdog + debounce + queue
│   ├── diagnose.py                 # Healthcheck (Ollama, ChromaDB, disk, orphans)
│   └── utils/
│       ├── hashing.py              # SHA-256
│       ├── locks.py                # Verrou par doc_id
│       └── secure_yaml.py          # Chargement .ragrules.yaml sécurisé
├── scripts/
│   ├── ingest_cli.py               # CLI standalone
│   └── tag_cli.py                  # CLI retaguer
├── rag_index/                      # Généré à l'exécution (gitignored)
├── tests/
│   ├── test_extractors.py
│   ├── test_chunker.py
│   ├── test_tagging_heuristics.py
│   ├── test_tagging_llm.py
│   ├── test_ollama_client.py
│   ├── test_watcher.py
│   ├── test_diagnose.py
│   └── test_server.py
├── pyproject.toml
├── .gitignore
└── README.md
```

**Changements vs v3 :**
- `model_manager.py` → supprimé (remplacé par `ollama_client.py`)
- `watcher/path_index.py` → intégré dans `storage.py`
- Ajout `ollama_client.py` (nouveau)
- Ajout `test_ollama_client.py` (nouveau)

---

## 15. Gestion des Erreurs et Edge Cases

- **Ollama injoignable** : fallback H1 pur pour le tagging, embedding impossible → ingestion mise en attente avec retry ou échec explicite
- **Modèle Ollama non trouvé** : erreur explicite avec suggestion `ollama pull <model>` ; si `auto_pull: true`, tentative de pull automatique
- **Timeout tagging H2** : document ingéré avec `llm_status: "timeout"` dans les métadonnées
- **Timeout embedding** : erreur bloquante pour le fichier concerné (pas de fallback possible sans embedding)
- **Fichier corrompu** : skip avec erreur listée dans `ingest_directory.errors`
- **ChromaDB locked** : retry avec backoff exponentiel (3 tentatives max)
- **Ingestion concurrente sur même doc_id** : coalescée par verrou applicatif
- **Suppression race condition** : fichier supprimé pendant ingestion → cleanup des chunks en attente
- **Fichier modifié pendant son ingestion** : debounce + verrou → ingestion redémarrée proprement après complétion
- **Tags invalides dans les requêtes** : `search_docs` retourne `{error: "Tag 'X' non trouvé", suggestions: [...]}`
- **Changement de modèle d'embedding** : erreur explicite avec suggestion `reindex_all`
- **Dépassement `max_chunks_per_doc`** : document ingéré avec warning, tronqué à la limite
- **Watcher sur dossier inaccessible** : erreur retournée à `watch_directory`, pas de crash silencieux

---

## 16. Sécurité

### 16.1 Validation de `.ragrules.yaml`

- Chargement via `yaml.safe_load` uniquement (pas de `yaml.load`)
- Taille maximale : `security.ragrules_max_bytes` (défaut 100 KB)
- Patterns glob : validés par `pathlib.PurePath.match`
- Regex utilisateur : compilés avec timeout (`regex` package, `timeout=0.2s`) pour éviter les attaques ReDoS

### 16.2 Isolation des chemins

- `dir_path` doit être absolu et existant
- Pas de symlinks suivis en dehors du dossier racine (validation canonique `Path.resolve()` + check `is_relative_to`)
- Refus silencieux des fichiers commençant par `.` sauf whitelist (`.md`)

### 16.3 Secrets et logs

- Aucun contenu de document n'est logué (seulement chemins et statuts)
- L'URL Ollama est loguée sans credentials (si auth nécessaire, les headers sont masqués)
- Les métadonnées loguées sont filtrées des champs sensibles

### 16.4 Communication Ollama

- La connexion vers Ollama utilise HTTP simple (réseau privé/localhost attendu)
- Si l'instance Ollama est exposée sur un réseau non fiable, l'utilisateur est responsable de la sécurisation (reverse proxy TLS, firewall)
- Aucun credential n'est envoyé par défaut ; un header `Authorization` optionnel est configurable via `ollama.auth_header`

---

## 17. Versioning & Migrations

### 17.1 Schéma

- Chaque chunk inclut `schema_version: "1.0"`
- Un fichier `rag_index/schema_version` indique la version majeure de la collection

### 17.2 Migrations

- Changement mineur (champ ajouté avec défaut) : rétrocompatible, mise à jour transparente à l'écriture
- Changement majeur (dimensions embedding, format metadata) : nécessite `reindex_all` explicite
- Les migrations sont tracées dans `rag_index/migrations.log`

### 17.3 Compatibilité des modèles

- Le modèle de tagging H2 versioned : tag cache invalidé si le modèle Ollama change (ex. `qwen2.5:3b` → `phi4-mini`)
- Le modèle d'embedding versioned : incompatibilité = `reindex_all` requis (bloquant)

---

## 18. Critères d'Acceptation (Definition of Done)

- [ ] Le serveur se lance en stdio via `python -m mcp_rag.server`
- [ ] **RAM serveur MCP < 400 MB en idle** (pas de modèles locaux)
- [ ] **Connexion Ollama fonctionnelle** : `diagnose` retourne `ollama.reachable: true`
- [ ] **Modèles Ollama requis** : `diagnose` confirme la présence de `embed_model`, `tag_model`, `vision_model`
- [ ] `ingest_directory` traite récursivement un dossier de 500+ fichiers mixtes sans crash
- [ ] Les PDF scannés et images sont correctement traités via le modèle vision Ollama
- [ ] Le tagging H1 applique les règles `.ragrules.yaml` et les regex de chemin
- [ ] Le tagging H2 produit des tags sémantiques valides (JSON structuré via Ollama JSON mode)
- [ ] Le cache de tags fonctionne (ré-ingestion fichier inchangé = 0 appel Ollama)
- [ ] `search_docs` filtre correctement par `tags_mode`
- [ ] `watch_directory` détecte créations, modifications et suppressions en temps réel
- [ ] Les suppressions filesystem sont répercutées dans ChromaDB (`sync_deletions=true`)
- [ ] `get_tags` retourne la taxonomie complète avec comptages exacts
- [ ] `diagnose` détecte correctement Ollama, orphans, espace disque, watchers
- [ ] `reindex_all` fonctionne sans perte de tags (recyclage cache)
- [ ] `unload_models` envoie correctement `keep_alive: 0` à Ollama
- [ ] `get_stats` reflète l'état réel de l'index
- [ ] L'URL Ollama est configurable via `config.yaml` et variables d'environnement
- [ ] Fallback H1-only si Ollama est injoignable (tagging dégradé, pas de crash)
- [ ] Connexion Hermes fonctionnelle via `~/.hermes/config.yaml`
- [ ] Tests unitaires > 80 % de couverture sur extractors/chunker/tagging/storage/ollama_client
- [ ] Timeout Ollama respecté (fallback H1 sans blocage MCP)
- [ ] Verrou par `doc_id` testé : 2 événements concurrents → une seule ingestion
- [ ] `.ragrules.yaml` malveillant (bombe YAML, regex catastrophique) → rejeté proprement

---

## 19. Prochaines Étapes

1. **Scaffolding** : `pyproject.toml`, structure de dossiers, config de base
2. **OllamaClient** : client HTTP async, healthcheck, embed/chat/vision (pierre angulaire, tester en premier)
3. **Core Ingestion** : `extractors.py` + `chunker.py` + `embeddings.py` (via Ollama) + `hashing.py`
4. **Tagging Engine** : `heuristics.py` (H1) + `llm_tagger.py` (H2 via Ollama `/api/chat` JSON mode, cache SQLite)
5. **Storage & MCP** : `storage.py` (ChromaDB + path_index SQLite) + `server.py` (15 outils)
6. **WatchFolder** : `fs_watcher.py` (watchdog, debounce, queue, verrous)
7. **Diagnose & Migration** : `diagnose.py` + `reindex_all` + schema versioning
8. **CLI** : `scripts/ingest_cli.py` + `scripts/tag_cli.py`
9. **Sécurité** : `secure_yaml.py` + validation chemins + regex timeout
10. **Tests & Validation** : Jeu de test hétérogène (50+ docs variés), mock Ollama, tests intégration
11. **Connexion Hermes** : Configuration `~/.hermes/config.yaml` + test E2E complet
12. **Documentation** : `README.md` avec quick start, config Ollama, tagging, watch folder, troubleshooting
