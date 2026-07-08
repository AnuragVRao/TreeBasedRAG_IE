import logging
import os

from pipeline import RaptorPipeline

# ── Logging ────────────────────────────────────────────────────────────────────
# Set to logging.DEBUG to see per-query strategy, MMR drops, and token counts
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Config ─────────────────────────────────────────────────────────────────────
PDF_PATH  = "./data/datafile.pdf"
TREE_PATH = "./my_tree"


def _tree_exists(tree_path: str) -> bool:
    return (
        os.path.exists(os.path.join(tree_path, "tree.json"))
        and os.path.exists(os.path.join(tree_path, "embeddings.npz"))
    )

def main() -> None:
    pipe = RaptorPipeline(
        embed_model         = "BAAI/bge-small-en-v1.5",
        llm_model           = "qwen3:8b",
        summary_model       = "qwen2.5:3b",
        max_tokens          = 3000,
        mmr_threshold       = 0.75,   # tighten to 0.80 if redundancy persists
        relevance_threshold = 0.20,   # lowered to 0.20 to ensure specific leaf nodes pass the filter
        layer_score_threshold= 0.0,   # lowered to 0.0 so traversal always follows the strongest available branches to the leaves
        enable_generation   = True,
    )

    if _tree_exists(TREE_PATH):
        print(f"Loading existing tree from '{TREE_PATH}'...")
        pipe.load(TREE_PATH)
    else:
        print(f"No tree found at '{TREE_PATH}'. Building from '{PDF_PATH}'...")
        pipe.build(pdf_path=PDF_PATH, save_path=TREE_PATH)

    print("\nReady. Ask a question (type 'exit' to quit).")
    print("Tip: prefix your query with 'collapsed:' or 'traversal:' to force a retrieval strategy.\n")

    while True:
        raw = input("Question: ").strip()

        if not raw:
            continue
        if raw.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        # ── Optional strategy override via prefix ──────────────────────────────
        # Typing "traversal: what are cutaneous receptors" forces traversal mode.
        # Useful for testing and comparing strategies side by side.
        force_strategy = None
        query = raw
        if raw.lower().startswith("collapsed:"):
            force_strategy = "collapsed"
            query = raw[len("collapsed:"):].strip()
        elif raw.lower().startswith("traversal:"):
            force_strategy = "traversal"
            query = raw[len("traversal:"):].strip()

        if not query:
            print("Empty query after prefix — please include a question.")
            continue

        # Set enable_generation=False to inspect raw retrieval context
        answer = pipe.query(query, force_strategy=force_strategy)

        # Log the classified query type here since testing is active
        classified_type = getattr(pipe, 'last_query_type', 'unknown')
        logger.info(f"Final pipeline query_type: '{classified_type}'")
        logger.info(f"Answer generated for: '{query[:60]}'")

        print("\n── Answer ────────────────────────────────────────────────────")
        print(answer)
        print("─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()