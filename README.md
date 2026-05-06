# MCP RAG Server

Serveur MCP local pour la recherche sémantique (RAG) avec auto-tagging, watch folder, et gestion mémoire lazy-load.

## Quick Start

```bash
# Installation
pip install -e .

# Configuration (optionnel)
cp config.example.yaml config.yaml

# Lancement du serveur MCP (stdio)
python -m mcp_rag.server

# Ingestion batch via CLI
python -m scripts.ingest_cli /chemin/vers/documents
```

## Architecture

- **Embeddings 100% locaux** : `sentence-transformers` (pas d'API cloud)
- **Tagging hybride** : règles heuristiques (H1) + LLM local petit modèle (H2)
- **Watch folder** : synchronisation temps réel avec le filesystem
- **RAM minimale en veille** : modèles chargés à la demande, déchargés après inactivité

## Connexion Hermes

Ajouter dans `~/.hermes/config.yaml` :

```yaml
mcp_servers:
  rag-docs:
    command: "python3"
    args: ["-m", "mcp_rag.server", "--config", "/abs/path/to/config.yaml"]
    timeout: 180
    connect_timeout: 60
```

## Outils MCP exposés

| Outil | Description |
|-------|-------------|
| `ingest_directory` | Indexe un dossier avec tagging auto |
| `search_docs` | Recherche sémantique filtrable par tags |
| `get_document` | Récupère chunks d'un document |
| `list_documents` | Liste paginée filtrable |
| `delete_document` | Supprime un document indexé |
| `clear_index` | Vide l'index |
| `watch_directory` | Active/désactive le watch folder |
| `get_tags` | Taxonomie complète |
| `tag_document` | Retaguer un fichier |
| `reindex_all` | Réindexe avec un nouveau modèle d'embedding |
| `diagnose` | Healthcheck complet |
| `get_model_status` | État RAM des modèles |
| `unload_models` | Force le déchargement |
| `preload_models` | Précharge les modèles |

## License

MIT
