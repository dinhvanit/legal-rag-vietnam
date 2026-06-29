"""
src/reference_extractor.py
Trích relevant_docs / relevant_articles từ danh sách context ĐÃ RERANK.

Tách riêng khỏi AnswerGenerator để pipeline retrieval-only (fast_retrieval.py — KHÔNG nạp LLM)
tái dùng được CÙNG một logic, tránh lệch code giữa hai đường.

Chiến lược: lấy TOP-N theo thứ tự rerank (Settings.RELEVANT_ARTICLES_MAX / RELEVANT_DOCS_MAX).
Bằng chứng (mô phỏng 50 câu GT): top-2 rerank cho F2 cao nhất; giao với citation LLM làm giảm F2.
"""
import json
import logging
from typing import List, Dict, Tuple

from config.settings import Settings

logger = logging.getLogger(__name__)


def load_manifest(path: str = Settings.LAW_MANIFEST_PATH) -> Dict:
    """Nạp law_manifest.json (dict keyed by số hiệu văn bản)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Không tìm thấy law_manifest.json tại {path}. Chạy không có manifest.")
        return {}


def canonical_doc_string(doc_number: str, fallback_title: str, manifest: Dict) -> str:
    """
    Tra law_manifest.json để lấy chuỗi chuẩn "<Số hiệu>|<Tên văn bản>" theo format BTC.
    Ưu tiên field "btc_standard_string"; fallback sang title của chunk khi không có trong manifest.

    AUDIT: nếu "btc_standard_string" trong manifest không CHỨA đúng doc_number đang tra
    (tức entry bị gán/chuẩn hoá lệch số hiệu - nghi vấn gốc rễ lỗi "trích dẫn thừa văn
    bản không liên quan" như case Nghị quyết 98/2023/QH15 bị lẫn vào câu hỏi về
    04/2017/QH14), log WARNING ngay tại đây để phát hiện sớm trước khi chạy full 2000 câu.
    """
    entry = manifest.get(doc_number)
    if isinstance(entry, dict) and entry.get("btc_standard_string"):
        canonical = entry["btc_standard_string"]
        if doc_number not in canonical:
            logger.warning(
                f"[reference_extractor] NGHI VẤN LỖI MANIFEST: tra doc_number='{doc_number}' "
                f"nhưng btc_standard_string trả về không chứa số hiệu này -> '{canonical}'. "
                f"Kiểm tra lại law_manifest.json (có thể bị gán/copy lệch entry)."
            )
        return canonical
    if fallback_title:
        return f"{doc_number}|{fallback_title}"
    return f"{doc_number}|Văn bản {doc_number}"


def validate_manifest_consistency(manifest: Dict) -> List[str]:
    """
    Quét TOÀN BỘ law_manifest.json MỘT LẦN để phát hiện entry có "btc_standard_string"
    không chứa đúng key (doc_number) của nó — dấu hiệu manifest bị build lệch (vd 2 văn
    bản trùng key, hoặc copy-paste nhầm string chuẩn giữa các entry).

    Khuyến nghị: chạy hàm này NGAY SAU KHI load manifest, TRƯỚC khi chạy full 2000 câu,
    để phát hiện lỗi mapping mang tính HỆ THỐNG (ảnh hưởng nhiều câu, không chỉ 1 case).

    Trả về list message lỗi (rỗng nếu manifest sạch).
    """
    issues: List[str] = []
    for key, entry in manifest.items():
        if not isinstance(entry, dict):
            continue
        canonical = entry.get("btc_standard_string", "")
        if canonical and key not in canonical:
            issues.append(
                f"Key='{key}' nhưng btc_standard_string='{canonical}' KHÔNG chứa key này."
            )
    if issues:
        logger.warning(
            f"[validate_manifest_consistency] Phát hiện {len(issues)} entry nghi vấn lệch mapping:\n"
            + "\n".join(f"  - {m}" for m in issues[:30])
            + (f"\n  ... và {len(issues) - 30} lỗi khác" if len(issues) > 30 else "")
        )
    else:
        logger.info("[validate_manifest_consistency] Manifest sạch, không phát hiện lệch mapping.")
    return issues


def extract_references_topn(contexts: List[Dict], manifest: Dict) -> Tuple[List[str], List[str]]:
    """
    Lấy TOP-N văn bản/Điều đầu tiên theo thứ tự rerank.
    Yêu cầu: `contexts` giữ nguyên thứ tự rerank (điểm cao -> thấp).
    Định dạng:
    - relevant_docs: ["mã văn bản|tên văn bản"]
    - relevant_articles: ["mã văn bản|tên văn bản|Điều X"]
    """
    relevant_docs: List[str] = []
    relevant_articles: List[str] = []
    seen_docs = set()
    seen_articles = set()

    for doc in contexts:
        metadata = doc.get("metadata", {})
        doc_number = metadata.get("doc_number", "").strip()
        doc_title = metadata.get("title", "").strip()
        article_id = metadata.get("article_id", "").strip()  # Ví dụ: "Điều 4"

        if not doc_number:
            continue

        cdoc = canonical_doc_string(doc_number, doc_title, manifest)

        if cdoc not in seen_docs and len(relevant_docs) < Settings.RELEVANT_DOCS_MAX:
            seen_docs.add(cdoc)
            relevant_docs.append(cdoc)

        if article_id:
            astr = f"{cdoc}|{article_id}"
            if astr not in seen_articles and len(relevant_articles) < Settings.RELEVANT_ARTICLES_MAX:
                seen_articles.add(astr)
                relevant_articles.append(astr)

        if (len(relevant_docs) >= Settings.RELEVANT_DOCS_MAX
                and len(relevant_articles) >= Settings.RELEVANT_ARTICLES_MAX):
            break

    return relevant_docs, relevant_articles


def extract_references_all(contexts: List[Dict], manifest: Dict) -> Tuple[List[str], List[str]]:
    """
    Lấy TẤT CẢ văn bản/Điều phân biệt từ `contexts` (KHÔNG cap theo Settings).
    Dùng cho luồng LLM-select số lượng biến thiên: `contexts` ở đây đã là tập LLM chọn ra
    (đã giới hạn bởi max_select), nên chỉ cần chuyển thành chuỗi chuẩn + loại trùng.
    Giữ nguyên thứ tự đầu vào (= thứ tự ưu tiên LLM chọn).
    """
    relevant_docs: List[str] = []
    relevant_articles: List[str] = []
    seen_docs = set()
    seen_articles = set()

    for doc in contexts:
        metadata = doc.get("metadata", {})
        doc_number = metadata.get("doc_number", "").strip()
        doc_title = metadata.get("title", "").strip()
        article_id = metadata.get("article_id", "").strip()
        if not doc_number:
            continue
        cdoc = canonical_doc_string(doc_number, doc_title, manifest)
        if cdoc not in seen_docs:
            seen_docs.add(cdoc)
            relevant_docs.append(cdoc)
        if article_id:
            astr = f"{cdoc}|{article_id}"
            if astr not in seen_articles:
                seen_articles.add(astr)
                relevant_articles.append(astr)

    return relevant_docs, relevant_articles


def extract_references_all_v2(contexts: List[Dict], manifest: Dict) -> Tuple[List[str], List[str]]:
    """
    Alias rõ nghĩa của extract_references_all(), dùng riêng cho luồng
    answer_intersect_v2.intersect_select_v2() — logic HOÀN TOÀN GIỐNG
    extract_references_all (không cap, giữ thứ tự, loại trùng), chỉ tách
    tên hàm để dễ trace trong notebook/log khi so sánh A/B với bản gốc.
    """
    return extract_references_all(contexts, manifest)