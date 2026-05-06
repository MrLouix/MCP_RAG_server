# Spécification : Serveur MCP RAG Local — v3 (Lazy-Load + Auto-Tagging + Watch Folder)

## 1. Objectif

Créer un serveur MCP (Model Context Protocol) auto-hébergé qui expose une base de connaissances RAG à l'agent Hermes. Le système ingère des documents hétérogènes (PDF, images, texte brut, Markdown, DOCX, CSV/Excel), les **tague automatiquement** (heuristiques + petit LLM local), les indexe sémantiquement avec des embeddings locaux, et sert des recherches **filtrables par tags** via des outils MCP stdio. Un **watcher de dossier** synchronise l'index en temps réel avec le système de fichiers.

**Contraintes fortes :**
- **100 % local** : aucune donnée ni requête ne quitte la machine
- **Embeddings et tagging locaux** : pas d'API cloud (OpenAI, Anthropic, etc.)
- **Empreinte RAM minimale en veille** : les modèles lourds (LLM de tagging, embedder, OCR) sont chargés *à la demande* et déchargés après une période d'inactivité configurable
- Capable d'ingérer des centaines de fichiers en une seule commande sans bloquer le serveur MCP
- Métadonnées strictes et traçables (source, page, date, hash, tags système et sémantiques)
- L'index reflète toujours l'état du dossier surveillé (ajout/suppression automatiques)
- **Cohérence du modèle d'embedding** : figé par collection, migration explicite

---

## 2. Architecture Globale

```
┌─────────────────┐     ┌──────────────────────────────────┐     ┌─────────────────┐
│  Dossier Source │────▶│ Pipeline Ingest + Tagging Engine │────▶│  ChromaDB Local │
│  (PDF/IMG/TXT)  │     │  H1:Règles → H2:LLM local       │     │  Persistent     │
│  ← sync auto ←  │◀────│  Watchdog (fs events queue)     │     │  (./rag_index/) │
│                 │     │  Lazy Model Manager (TTL)       │     │                 │
│  ┌─.ragrules.yaml│     └────────────────┬─────────────────┘     └────────┬────────┘
│  └─tag_cache.db │                      │                               │
└─────────────────┘          ┌───────────▼──────────┐                    │ stdio
                             │  Serveur FastMCP     │◀───────────────────┤
                             │  (mcp_rag)           │                    │
                             │  + Model Manager     │────────────────────┤
                             │  + Diagnose tools    │                    │
                             └──────────┬───────────┘                    │
                                        │                                 │
                                        ▼                                 │
                                ┌──────────────┐                          │
                                │ Hermes Agent │◀─────────────────────────┘
                                │ (Client MCP) │
                                └──────────────┘
```

**Flux nominal :**
1. L'utilisateur place ses docs dans un dossier source → le **watcher** les détecte
2. Le `ModelManager` **charge à la demande** extractor OCR + embedder + LLM de tagging
3. Le pipeline tourne : extraction → H1 → H2 → chunking → embedding → stockage
4. Après `idle_ttl` d'inactivité, les modèles sont **déchargés** (RAM libérée)
5. L'agent interroge via `search_docs` → l'embedder est rechargé (lazy) si besoin → résultats filtrés
6. Les suppressions filesystem sont répercutées automatiquement si `sync_deletions=true`

---

## 3. Pipeline d'Ingestion Batch

### 3.1 Extraction par format

| Format   | Bibliothèque             | Stratégie                                                                 |
|----------|--------------------------|---------------------------------------------------------------------------|
| PDF texte| `pdfplumber` / `PyMuPDF` | Extraction texte page par page, conservation structure                    |
| PDF scanné | `PyMuPDF` + `EasyOCR`  | Détection pages sans texte → OCR page complète → reconstruction           |
| Images   | `Pillow` + `EasyOCR`     | OCR multilingue (fra/eng) avec détection automatique de langue            |
| Markdown | `pathlib` / `markdown`   | Parsing natif, conservation des liens internes et blocs de code           |
| TXT      | `pathlib`                | Lecture directe, détection encodage (UTF-8 fallback Latin-1)              |
| DOCX     | `python-docx`            | Extraction texte des paragraphes, tableaux, styles                        |
| CSV/Excel| `pandas` / `openpyxl`    | Lecture structurée, concaténation des colonnes pertinentes                |

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

Le tagging est **hybride** et produit deux familles de tags : les tags *système* (déterministes) et les tags *sémantiques* (inférés par un modèle de langage local).

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

### 4.2 Couche H2 — Tags Sémantiques (LLM Local)

- **Modèles supportés** : `Qwen2.5-3B-Instruct`, `Phi-4-mini` (3.8B), `Gemma-2-2B-IT`
- **Format** : `GGUF` quantifié `Q4_K_M` (~1.5 – 2.2 Go)
- **Runtime** : `llama.cpp` via bindings Python (`llama-cpp-python`)
- **Mode de sortie** : **JSON Mode** avec `json_schema` imposé (GBNF grammar) — pas de sortie libre
- **Concurrence** : **un seul worker LLM dédié** avec queue FIFO (`llama-cpp-python` n'est pas thread-safe ; un pool multiplierait la RAM)
- **Lazy-loading** : le modèle est chargé à la première demande puis déchargé après `idle_ttl_seconds` (voir §7)

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
- Temperature 0.1, seed=42.

Document:
--- Début ---
{extrait_1500_tokens}
--- Fin ---
```

**Contrainte JSON Schema (GBNF)** : le `json_schema` passé à `llama.cpp` utilise les `enum` de la taxonomy pour forcer des valeurs valides.

### 4.3 Cache de Tags

Base SQLite locale `.rag_tag_cache.db` (préférée à JSONL pour la concurrence lecture/écriture) :
- Mapping `content_hash (SHA-256) → {tags_json, model_version, inferred_at}`
- Si un fichier est ré-ingéré et que son hash n'a pas changé, les tags H2 sont recyclés instantanément
- Invalidation automatique si la version du modèle de tagging change

### 4.4 Fallback et Robustesse

- Si le LLM local n'est pas chargé ou timeout (défaut **10 s**, configurable) → ingestion se poursuit avec uniquement les tags H1
- Le timeout est strict : pas de blocage du serveur MCP stdio
- Un warning est logué avec le chemin du fichier ayant échoué au tagging H2 (`llm_status: "timeout"` dans les métadonnées)

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
  "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
  "orphaned": false,
  "tags": {
    "system": ["client:acme", "year:2026", "type:facture", "format:pdf"],
    "semantic": ["domaine:financier", "priorite:normal", "langue:fr", "entite:tv_a_credit"],
    "model": "Qwen2.5-3B-Instruct-Q4_K_M",
    "inferred_at": "2026-05-06T14:30:01Z",
    "llm_status": "ok"
  }
}
```

**Notes :**
- `schema_version` permet de futures migrations sans casser l'index
- `embedding_model` est répliqué à chaque chunk : on peut détecter une incohérence si l'index est partagé
- `llm_status ∈ {"ok", "timeout", "error", "disabled"}`
- Les **tags sont calculés au niveau documentaire** mais **répliqués dans chaque chunk**, permettant aux filtres `where` du vector store de fonctionner même sur un fragment isolé
- Le champ `confidence` est **retiré** de cette version (les log-probs de `llama.cpp` ne sont pas exposés de façon stable par `llama-cpp-python` ; réintroduction possible en v2 avec `logit_bias` + post-traitement)

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
    │                           │ (LLM local)  │      ▼
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
                                               ┌──────────────┐
                                               │ Embedding    │
                                               │ (batch 32)   │
                                               └──────────────┘
                                                        │
                                                        ▼
                                               ┌──────────────┐
                                               │ Write Chroma │
                                               │ (bulk insert)│
                                               └──────────────┘
```

**Sérialisation par `doc_id` :** un verrou applicatif (lock en mémoire, keyed par `doc_id`) empêche l'ingestion concurrente d'un même fichier (cas : watcher `modified` qui arrive pendant une ingestion batch). Les évts en double sur le même `doc_id` sont coalescés.

---

## 7. Gestion Mémoire : Lazy-Load & Idle-Unload (requis central)

### 7.1 Principe

Le serveur ne doit consommer **que ~300–500 MB en veille** (Python + ChromaDB). Tous les modèles lourds sont chargés *à la demande* via un `ModelManager` central, et **déchargés automatiquement** après une période d'inactivité.

### 7.2 Composants gérés

| Composant | RAM chargé | Déclencheur de chargement | Libéré après |
|-----------|-----------|---------------------------|---------------|
| **Embedder** (`sentence-transformers`) | ~450 MB | `ingest_directory`, `search_docs`, `tag_document` | `idle_ttl_embedder` (défaut 300 s) |
| **LLM tagger** (`llama.cpp`) | ~2.2 GB | `ingest_directory` (auto_tag=true), `tag_document` | `idle_ttl_llm` (défaut 120 s) |
| **OCR** (`EasyOCR`) | ~1.3 GB | Extraction PDF scanné ou image | `idle_ttl_ocr` (défaut 180 s) |
| **ChromaDB client** | ~50 MB | Startup | Jamais (léger) |

### 7.3 API interne `ModelManager`

```python
class ModelManager:
    def get_embedder(self) -> SentenceTransformer:
        """Charge si nécessaire, reset le timer d'idle."""
    def get_llm(self) -> Llama:
        """Idem, pour le tagger LLM."""
    def get_ocr(self) -> easyocr.Reader:
        """Idem, pour EasyOCR."""
    def unload_if_idle(self) -> dict:
        """Appelé par un thread de garbage collection toutes les 30s."""
    def unload_all(self, force: bool = False) -> dict:
        """Déchargement explicite via outil MCP."""
    def get_status(self) -> dict:
        """{embedder: {loaded, last_used, ram_mb}, llm: {...}, ocr: {...}}"""
```

### 7.4 Thread de surveillance

Un thread daemon dédié (`model_gc`) tourne en continu :
- Tick toutes les 30 s
- Pour chaque modèle : `if now - last_used > idle_ttl: unload()`
- Appel explicite de `gc.collect()` + `torch.cuda.empty_cache()` (si GPU) après unload
- Logs structurés : `model_unloaded`, `ram_freed_mb`, `duration_idle_s`

### 7.5 Outils MCP associés

- `get_model_status` : retourne l'état et la RAM utilisée par chaque modèle
- `unload_models` : force le déchargement immédiat (utile après un gros batch avant une longue pause)
- `preload_models` : charge explicitement un ou plusieurs modèles (utile avant un batch volumineux pour éviter le coût du premier chargement)

### 7.6 Configuration

```yaml
memory:
  lazy_load: true
  idle_ttl_embedder: 300    # secondes
  idle_ttl_llm: 120
  idle_ttl_ocr: 180
  gc_tick_seconds: 30
  aggressive_gc: true       # gc.collect() + malloc_trim après unload
```

### 7.7 Compromis et trade-offs

- **Pros** : empreinte mémoire idle minimale (< 500 MB), multi-tenant possible
- **Cons** : première requête après idle = latence de chargement (5-15 s pour le LLM, 2-4 s pour l'embedder, 3-6 s pour OCR)
- **Mitigation** : `preload_models` avant un batch connu, ou TTL très long (ex. 3600 s) si RAM disponible

---

## 8. Synchronisation Temps Réel (Watch Folder)

### 8.1 Filesystem Watcher

- **Librairie** : `watchdog` (cross-platform, abstraction de `inotify` / `ReadDirectoryChangesW` / `FSEvents`)
- **Événements surveillés** : `created`, `modified`, `moved`, `deleted`
- **Debouncing** : fenêtre de `debounce_ms` (défaut 2000 ms) pour éviter d'ingérer un fichier en cours d'écriture
- **Queue asynchrone** : les événements sont mis en file et traités par un pool de workers (**extraction/embedding parallélisables**, tagging H2 strictement sérialisé via le worker LLM unique)

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

### 9.1 Modèle d'Embedding (défaut 100 % local)

- **Modèle principal** : `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - ~420 MB, multilingue (optimisé français/anglais), dimensions **384**
  - Téléchargé une fois, mis en cache localement (`~/.cache/huggingface/`)
- **Fallback léger** : `all-MiniLM-L6-v2` (80 MB, 384 dims) pour RAM contrainte < 4 Go
- **Option avancée (facultative)** : `nomic-embed-text-v2` via Ollama (768 dims, contexte 8192 tokens)
  - Nécessite Ollama installé localement (port 11434)
  - **Non recommandé** par défaut : ajoute une dépendance système
  - Activable via config explicite `embedding_backend: ollama`

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
- Outil `reindex_all(new_embedding_model, confirm=true)` : vide la collection, reparcourt tous les fichiers listés dans l'index inversé, réembedded avec le nouveau modèle
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
| **`reindex_all`** | Réindexe tous les docs avec nouveau modèle | `new_embedding_model`, `confirm`, `workspace` | `{status, reindexed, duration_ms}` |
| **`diagnose`** | Healthcheck complet | `workspace` (optionnel) | `{models, disk, chroma, watchers, orphans, warnings}` |
| **`get_model_status`** | État des modèles chargés en RAM | *(aucun)* | `{embedder, llm, ocr}` (chacun avec `loaded`, `ram_mb`, `last_used`) |
| **`unload_models`** | Force le déchargement des modèles | `models: list[str]` (défaut: tous) | `{unloaded, ram_freed_mb}` |
| **`preload_models`** | Précharge des modèles avant un batch | `models: list[str]` | `{loaded, ram_used_mb}` |

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
    "model_path": "/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    "n_ctx": 4096,
    "n_threads": 4,
    "taxonomy": {
      "domaine": ["financier", "juridique", "technique", "commercial", "rh"],
      "priorite": ["urgent", "normal", "faible"],
      "confidentialite": ["public", "interne", "confidentiel"]
    },
    "timeout_ms": 10000,
    "use_cache": true
  },
  "tag_rules_path": "/chemin/.ragrules.yaml",
  "unload_after": false
}
```

`unload_after: true` → libère les modèles à la fin du batch (utile pour les gros lots ponctuels).

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
  "duration_ms": 12450,
  "ram_peak_mb": 4200
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

### 11.7 `reindex_all` (nouveau)

```json
{
  "new_embedding_model": "paraphrase-multilingual-mpnet-base-v2",
  "confirm": true,
  "workspace": "default"
}
```

**Retour :**
```json
{
  "status": "success",
  "previous_model": "paraphrase-multilingual-MiniLM-L12-v2",
  "new_model": "paraphrase-multilingual-mpnet-base-v2",
  "reindexed": 142,
  "tag_cache_hits": 142,
  "llm_inferences": 0,
  "duration_ms": 184300
}
```

### 11.8 `diagnose` (nouveau)

Healthcheck opérationnel. Ne charge **aucun modèle** (lecture métadonnées uniquement).

**Retour :**
```json
{
  "status": "ok",
  "chroma": { "reachable": true, "collections": ["default", "perso"] },
  "disk": { "index_size_mb": 842, "free_gb": 128, "warning": null },
  "models": {
    "embedder": { "installed": true, "loaded": false, "path": "~/.cache/huggingface/.../" },
    "llm": { "installed": true, "loaded": false, "path": "/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf" },
    "ocr": { "installed": true, "loaded": false }
  },
  "watchers": [{ "id": "wd_01", "path": "/data/docs", "active": true, "queue_size": 0 }],
  "orphans": { "count": 3, "examples": ["/data/docs/obsolete.pdf"] },
  "warnings": ["3 documents orphans détectés"]
}
```

### 11.9 `get_model_status` / `unload_models` / `preload_models`

```json
// get_model_status : aucune entrée
{
  "embedder": { "loaded": true, "ram_mb": 450, "last_used_s_ago": 12 },
  "llm": { "loaded": false, "ram_mb": 0, "last_used_s_ago": null },
  "ocr": { "loaded": false, "ram_mb": 0, "last_used_s_ago": null }
}

// unload_models
{ "models": ["embedder", "llm"] }
// → { "unloaded": ["embedder", "llm"], "ram_freed_mb": 2650 }

// preload_models
{ "models": ["embedder", "llm"] }
// → { "loaded": ["embedder", "llm"], "ram_used_mb": 2650 }
```

---

## 12. Configuration & Paramétrage

Fichier `config.yaml` :

```yaml
rag:
  index_path: "./rag_index"
  embedding_model: "paraphrase-multilingual-MiniLM-L12-v2"
  embedding_backend: "sentence-transformers"   # ou "ollama"
  embedding_fallback: "all-MiniLM-L6-v2"
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

memory:
  lazy_load: true
  idle_ttl_embedder: 300
  idle_ttl_llm: 120
  idle_ttl_ocr: 180
  gc_tick_seconds: 30
  aggressive_gc: true

tagging:
  auto_tag_enabled: true
  model_path: "/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
  n_ctx: 4096
  n_threads: 4
  timeout_ms: 10000
  use_cache: true
  cache_path: ".rag_tag_cache.db"
  taxonomy:
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

---

## 13. Contraintes & Performance

| Aspect | Cible | Note |
|--------|-------|------|
| **RAM idle (modèles déchargés)** | **< 500 MB** | Python + ChromaDB client + watchers uniquement |
| **RAM en charge (ingest+tag+OCR)** | 5 – 6 GB | Embedder (450) + LLM (2200) + OCR (1300) + Chroma (~200) + buffers (~1000) |
| **RAM en charge (ingest sans OCR)** | 3 – 4 GB | Sans EasyOCR |
| **RAM en recherche seule** | ~1 GB | Embedder chargé uniquement |
| **Première requête après idle** | +5 à 15 s | Rechargement des modèles (cf §7.7) |
| **Ingestion batch** | 300 docs / 15 – 25 min | Parallélisation extraction/embed, tagging H2 sérialisé |
| **Recherche (chaud)** | < 800 ms (top_k=5) | Embedder déjà en RAM |
| **Tagging H2** | 500 – 2000 ms / doc | Qwen2.5-3B Q4_K_M, CPU 4 threads |
| **Watch folder latence** | < 3 s | Debounce 2s + traitement |
| **Portabilité** | Linux / macOS / WSL | Dépendances Python |
| **Reproductibilité** | Hash SHA-256, seed LLM fixe | Détection doublons, tagging déterministe |

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
│   ├── model_manager.py            # Lazy-load + idle-unload (§7)
│   ├── ingest.py                   # Pipeline extraction/chunking/embedding
│   ├── extractors.py               # Lecteurs par format (PDF, IMG, TXT, MD, DOCX, CSV)
│   ├── chunker.py                  # Stratégie de segmentation
│   ├── embeddings.py               # Wrapper sentence-transformers / Ollama
│   ├── storage.py                  # Abstraction ChromaDB + index inversé
│   ├── config.py                   # Pydantic/YAML configuration
│   ├── tagging/
│   │   ├── __init__.py
│   │   ├── engine.py               # Orchestrateur H1 + H2
│   │   ├── heuristics.py           # H1 : règles, regex, .ragrules.yaml
│   │   ├── llm_tagger.py           # H2 : llama.cpp, GBNF, cache SQLite
│   │   └── taxonomy.py             # Définition et validation
│   ├── watcher/
│   │   ├── __init__.py
│   │   ├── fs_watcher.py           # watchdog + debounce + queue
│   │   └── path_index.py           # Index inversé SQLite
│   ├── diagnose.py                 # Healthcheck et orphan detection
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
│   ├── test_watcher.py
│   ├── test_model_manager.py
│   ├── test_diagnose.py
│   └── test_server.py
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## 15. Gestion des Erreurs et Edge Cases

- **LLM non disponible** : fallback H1 pur, warning logué, ingestion continue
- **Timeout tagging H2** : document ingéré avec `llm_status: "timeout"` dans les métadonnées
- **Fichier corrompu** : skip avec erreur listée dans `ingest_directory.errors`
- **ChromaDB locked** : retry avec backoff exponentiel (3 tentatives max)
- **Ingestion concurrente sur même doc_id** : coalescée par verrou applicatif
- **Suppression race condition** : fichier supprimé pendant ingestion → cleanup des chunks en attente
- **Fichier modifié pendant son ingestion** : debounce + verrou → ingestion redémarrée proprement après complétion
- **Tags invalides dans les requêtes** : `search_docs` retourne `{error: "Tag 'X' non trouvé", suggestions: [...]}`
- **Changement de modèle d'embedding** : erreur explicite avec suggestion `reindex_all`
- **Dépassement `max_chunks_per_doc`** : document ingéré avec warning, tronqué à la limite
- **Mémoire insuffisante au chargement** : fallback sur modèle léger + warning, ou échec explicite si même le fallback ne passe pas
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
- Les métadonnées loguées sont filtrées des champs sensibles (config de tagging peut masquer certaines clés)

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

- Le modèle de tagging H2 versioned : tag cache invalidé si la version change (ex. `Qwen2.5-3B-v1` → `Qwen2.5-3B-v2`)
- Le modèle d'embedding versioned : incompatibilité = `reindex_all` requis (bloquant)

---

## 18. Critères d'Acceptation (Definition of Done)

- [ ] Le serveur se lance en stdio via `python -m mcp_rag.server`
- [ ] **RAM idle < 500 MB après 5 min sans activité** (tous modèles déchargés)
- [ ] **RAM en charge atteint la cible (3-6 GB selon composants actifs)**
- [ ] **Rechargement lazy fonctionnel** : première requête après idle → modèles rechargés automatiquement
- [ ] `ingest_directory` traite récursivement un dossier de 500+ fichiers mixtes sans crash
- [ ] Les PDF scannés et images sont correctement passés à l'OCR
- [ ] Le tagging H1 applique les règles `.ragrules.yaml` et les regex de chemin
- [ ] Le tagging H2 produit des tags sémantiques valides (JSON structuré, pas d'hallucination)
- [ ] Le cache de tags fonctionne (ré-ingestion fichier inchangé = 0 appel LLM)
- [ ] `search_docs` filtre correctement par `tags_mode`
- [ ] `watch_directory` détecte créations, modifications et suppressions en temps réel
- [ ] Les suppressions filesystem sont répercutées dans ChromaDB (`sync_deletions=true`)
- [ ] `get_tags` retourne la taxonomie complète avec comptages exacts
- [ ] `diagnose` détecte correctement orphans, espace disque, watchers
- [ ] `reindex_all` fonctionne sans perte de tags (recyclage cache)
- [ ] `unload_models` libère effectivement la RAM (vérif via `psutil`)
- [ ] `get_stats` reflète l'état réel de l'index
- [ ] Zéro appel réseau vers des services externes
- [ ] Connexion Hermes fonctionnelle via `~/.hermes/config.yaml`
- [ ] Tests unitaires > 80 % de couverture sur extractors/chunker/tagging/storage/model_manager
- [ ] Timeout tagging H2 respecté (fallback H1 sans blocage MCP)
- [ ] Verrou par `doc_id` testé : 2 événements concurrents → une seule ingestion
- [ ] `.ragrules.yaml` malveillant (bombe YAML, regex catastrophique) → rejeté proprement

---

## 19. Prochaines Étapes

1. **Scaffolding** : `pyproject.toml`, structure de dossiers, config de base
2. **ModelManager** : lazy-load + idle-unload + thread GC (pierre angulaire, tester en premier)
3. **Core Ingestion** : `extractors.py` + `chunker.py` + `embeddings.py` + `hashing.py`
4. **Tagging Engine** : `heuristics.py` (H1) + `llm_tagger.py` (H2, GBNF, cache SQLite)
5. **Storage & MCP** : `storage.py` (ChromaDB + path_index SQLite) + `server.py` (15 outils)
6. **WatchFolder** : `fs_watcher.py` (watchdog, debounce, queue, verrous)
7. **Diagnose & Migration** : `diagnose.py` + `reindex_all` + schema versioning
8. **CLI** : `scripts/ingest_cli.py` + `scripts/tag_cli.py`
9. **Sécurité** : `secure_yaml.py` + validation chemins + regex timeout
10. **Tests & Validation** : Jeu de test hétérogène (50+ docs variés), mock LLM, tests RAM avec `psutil`
11. **Connexion Hermes** : Configuration `~/.hermes/config.yaml` + test E2E complet
12. **Documentation** : `README.md` avec quick start, config tagging, watch folder, troubleshooting
