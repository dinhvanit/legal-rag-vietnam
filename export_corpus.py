import json
import os
from qdrant_client import QdrantClient
from config.settings import Settings

def export_from_qdrant():
    print("--- Đang kết nối Qdrant Cloud để lấy dữ liệu làm Corpus BM25 ---")
    client = QdrantClient(
        url=Settings.QDRANT_URL,
        api_key=Settings.QDRANT_API_KEY
    )
    
    corpus = []
    # Dùng tính năng scroll để kéo toàn bộ data về (mỗi lần 1000 bản ghi)
    records, next_page_id = client.scroll(
        collection_name=Settings.COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False # Không lấy vector để file nhẹ, chỉ lấy text/metadata
    )
    
    while records:
        for record in records:
            payload = record.payload
            corpus.append({
                "id": str(record.id),
                "text": payload.get("text", ""), # Trường chứa nội dung văn bản
                "metadata": payload.get("metadata", {})
            })
        
        if not next_page_id:
            break
            
        records, next_page_id = client.scroll(
            collection_name=Settings.COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False,
            next_page_id=next_page_id
        )

    # Ghi ra file data/corpus_clean.json
    os.makedirs("data", exist_ok=True)
    output_path = "data/corpus_clean.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Hoàn thành! Đã tạo thành công file {output_path} với {len(corpus)} văn bản đoạn luật.")

if __name__ == "__main__":
    export_from_qdrant()