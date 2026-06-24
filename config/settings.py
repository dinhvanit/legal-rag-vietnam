"""
config/settings.py
Cấu hình tập trung cho toàn bộ pipeline RAG pháp lý.
Tối ưu cho môi trường Kaggle GPU (T4/P100) + DeepSeek-R1-Distill-Qwen-14B
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # =========================================================
    # QDRANT CLOUD
    # =========================================================
    QDRANT_URL = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME = "vietnamese_laws"
    EMBEDDING_DIM = 1024  # Chiều vector của AITeamVN/Vietnamese_Embedding

    # =========================================================
    # EMBEDDING MODEL
    # =========================================================
    EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"

    # =========================================================
    # BM25 & HYBRID RETRIEVAL
    # =========================================================
    RRF_K = 60          # Hằng số RRF chuẩn
    TOP_K_RAW = 30      # Số văn bản thô lấy từ mỗi nhánh (Dense + Sparse)

    # =========================================================
    # MULTI-QUERY EXPANSION
    # =========================================================
    NUM_QUERY_VARIANTS = 3   # Số biến thể câu hỏi sinh ra để mở rộng retrieval

    # =========================================================
    # RERANKER (Cross-Encoder)
    # Chạy local trên Kaggle GPU
    # =========================================================
    RERANKER_MODEL = "BAAI/bge-reranker-large"
    RERANKER_THRESHOLD = 0.30  # Ngưỡng tối thiểu để giữ văn bản (giảm xuống 0.30 để an toàn hơn)
    TOP_K_FINAL = 7            # Số văn bản đưa vào LLM (7 để có đủ context)

    # =========================================================
    # LLM - DeepSeek-R1-Distill-Qwen-14B trên Kaggle
    # =========================================================
    LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    LLM_MAX_NEW_TOKENS = 1024
    LLM_TEMPERATURE = 0.1      # Thấp → tránh ảo giác pháp lý
    LLM_TOP_P = 0.85
    LLM_REPETITION_PENALTY = 1.1

    # Tham số Regenerate khi Self-Verification thất bại
    LLM_REGEN_TEMPERATURE = 0.05   # Thấp hơn nữa khi sinh lại
    LLM_MAX_RETRIES = 1             # Số lần thử lại tối đa

    # =========================================================
    # PATHS
    # =========================================================
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    LAW_MANIFEST_PATH = os.path.join(DATA_DIR, "law_manifest.json")
    LEGAL_CATALOG_PATH = os.path.join(DATA_DIR, "legal_documents_catalog.json")
    BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_corpus.pkl")

    LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
    RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")
    EVAL_REPORT_PATH = os.path.join(LOG_DIR, "evaluation_report.md")

    # =========================================================
    # SELF-VERIFICATION RULES
    # =========================================================
    MIN_ARTICLE_REFS = 1      # Bắt buộc câu trả lời phải có ít nhất 1 tham chiếu Điều X
    MAX_CONTEXT_CHARS = 4000  # Giới hạn ký tự context nạp vào LLM (tránh vượt context window)
