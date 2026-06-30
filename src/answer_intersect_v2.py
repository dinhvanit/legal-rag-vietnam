"""
src/answer_intersect_v2.py  (BẢN SỬA LỖI của answer_intersect.py)

Sửa 2 lỗi đã xác nhận qua phân tích thực tế (luật sư review câu 1, câu 2):

LỖI 1 (case câu 1 — "thừa"/dư trích dẫn không liên quan):
    answer_intersect.py gốc dùng 2 SET RỜI RẠC (cited_arts, cited_docs) để match,
    không gắn theo CẶP (Điều, văn bản) như chúng thực sự đứng cạnh nhau trong câu.
    Khi answer chỉ cite đúng 1 căn cứ (vd "Điều 12 - 04/2017/QH14") mà pool không có
    chunk khớp ĐÚNG metadata (lệch định dạng doc_number/article_id giữa answer-text và
    metadata chunk) -> intersection rỗng -> code cũ FALLBACK "MÙ": lấy top-N theo
    rerank score, không xét nội dung có liên quan ngữ nghĩa hay không. Hệ quả: chèn
    thêm văn bản (vd Nghị quyết 98/2023/QH15 - cơ chế đặc thù TP.HCM) mà LLM không
    hề nhắc tới và không liên quan tới câu hỏi.

LỖI 2 (case câu 2 — "thiếu" căn cứ LLM đã cite đúng):
    Nếu pool[:pool_k] (rerank top-k) không chứa chunk khớp với Điều+văn bản mà LLM
    ĐÃ CITE ĐÚNG trong answer (vd "Điều 10 - Luật Đấu thầu 22/2023/QH15"), thì dù
    answer đúng 100%, intersect cũ vẫn không thể giữ lại được căn cứ đó vì nó tìm
    trong phạm vi pool hẹp.

CÁCH SỬA:
    1. Pairing theo VỊ TRÍ: ghép mỗi "Điều X" với số hiệu văn bản gần nó nhất về
       khoảng cách ký tự trong answer (tái dùng đúng kỹ thuật nearest-doc đã có ở
       post_processor.py::extract_legal_references, áp dụng cho luồng intersect).
    2. Bỏ fallback "mù lấy top theo rerank". Thay bằng 2 tầng:
       a. Match trong pool (ưu tiên, vì pool đã qua rerank -> tin cậy nhất).
       b. Nếu cặp (Điều, văn bản) LLM cite KHÔNG có trong pool nhưng CÓ THỰC trong
          toàn bộ corpus (corpus_lookup) -> vẫn lấy, vì đây là câu LLM cite đúng,
          chỉ là tầng retrieval/rerank xếp hạng thấp -> không nên bỏ qua.
       c. Chỉ khi một cặp cite hoàn toàn KHÔNG tồn tại trong corpus (rất có thể LLM
          bịa) -> loại bỏ, KHÔNG fallback thêm gì khác để tránh chèn nhiễu như lỗi 1.
       => Nếu sau (a)+(b) vẫn rỗng (answer không cite được gì hợp lệ) -> fallback
          top-1 duy nhất theo rerank (tối thiểu để answer không hoàn toàn vô căn cứ),
          KHÔNG lấy top-N như cũ (giảm thiểu rủi ro nhiễu).

Yêu cầu corpus_lookup: dict được build 1 lần từ corpus_clean.json, khóa theo
(article_number, doc_number) -> chunk đầy đủ, dùng để tra cứu tầng (b) phía trên.
"""
import re
import logging
from typing import List, Dict, Set, Tuple, Optional

from config.settings import Settings

logger = logging.getLogger(__name__)

_ARTICLE_NUM_RE = re.compile(r"[Đđ]iều\s+(\d+)")
# LƯU Ý QUAN TRỌNG: charset PHẢI có thêm "0-9" sau phần chữ, nếu không sẽ CẮT MẤT
# hậu tố số của ký hiệu loại văn bản (vd "QH14" -> chỉ bắt được "QH", mất "14";
# "NĐ-CP" thì không sao vì không có số ở cuối, nhưng "QH14", "TT01" sẽ bị lỗi).
# Đây là lỗi CÓ THẬT trong answer_intersect.py bản gốc (dùng [A-Za-zĐđ\-]+ thiếu
# 0-9) — là nguyên nhân match sai dẫn đến case "câu 1" sai lệch hoàn toàn khi debug
# bằng test_intersect_v2_repro.py. Pattern dưới đây đồng bộ với DOC_NUMBER_PATTERN
# đã chạy đúng trong post_processor.py.
_DOC_NUM_RE = re.compile(r"\d{1,4}/\d{4}/[A-ZĐa-zđ0-9\-]+", re.IGNORECASE)


def _article_num(article_id: str) -> str:
    m = re.search(r"(\d+)", article_id or "")
    return m.group(1) if m else ""


def parse_citation_pairs(answer_text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Bóc các CẶP (số Điều, số hiệu văn bản) theo VỊ TRÍ gần nhau trong câu,
    thay vì 2 set rời rạc không liên kết.

    Trả về list (article_number, doc_number_hoac_None), giữ thứ tự xuất hiện,
    loại trùng. doc_number = None nếu trong toàn câu answer không cite số hiệu
    nào cả (trường hợp hiếm, answer chỉ nói "Điều X" trần không kèm văn bản).
    """
    text = answer_text or ""
    article_events = [(m.start(), m.group(1)) for m in _ARTICLE_NUM_RE.finditer(text)]
    doc_events = [(m.start(), m.group(0)) for m in _DOC_NUM_RE.finditer(text)]

    pairs: List[Tuple[str, Optional[str]]] = []
    seen: Set[Tuple[str, Optional[str]]] = set()

    for pos, art_num in article_events:
        nearest_doc = None
        if doc_events:
            nearest_doc = min(doc_events, key=lambda e: abs(e[0] - pos))[1]
        key = (art_num, nearest_doc)
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    return pairs


def build_corpus_lookup(corpus: List[Dict]) -> Dict[Tuple[str, str], Dict]:
    """
    Build dict tra cứu (article_number, doc_number) -> chunk đầy đủ, dùng cho
    tầng fallback (b): khi answer cite đúng nhưng pool rerank không có ứng viên.

    `corpus`: list chunk dạng {"id", "text", "metadata": {"doc_number", "article_id", ...}}
    (cùng cấu trúc corpus_clean.json mà index_bm25.py / hybrid_retriever.py dùng).
    """
    lookup: Dict[Tuple[str, str], Dict] = {}
    for doc in corpus:
        md = doc.get("metadata", {}) or {}
        doc_number = (md.get("doc_number", "") or "").strip()
        art_num = _article_num(md.get("article_id", ""))
        if not doc_number or not art_num:
            continue
        key = (art_num, doc_number)
        # Giữ chunk đầu tiên gặp cho mỗi (article, doc) - tránh trùng lặp nếu
        # corpus có chia nhỏ nhiều khoản trong cùng 1 Điều.
        if key not in lookup:
            lookup[key] = doc
    logger.info(f"[answer_intersect_v2] corpus_lookup: {len(lookup)} cặp (Điều, văn bản) duy nhất.")
    return lookup


def intersect_select_v2(
    answer_text: str,
    ranked: List[Dict],
    corpus_lookup: Optional[Dict[Tuple[str, str], Dict]] = None,
    pool_k: int = 10,
    max_out: int = 5,
) -> List[Dict]:
    """
    Trả về list context được chọn theo cặp (Điều, văn bản) LLM ĐÃ CITE ĐÚNG,
    ưu tiên khớp trong pool rerank, fallback sang corpus_lookup nếu cặp đó tồn
    tại thật trong corpus nhưng rerank xếp hạng thấp/ngoài pool.

    KHÔNG fallback "mù lấy top-N rerank" khi citation không khớp gì cả — chỉ
    fallback top-1 duy nhất (tối thiểu để answer không hoàn toàn vô căn cứ),
    nhằm tránh chèn thêm văn bản không liên quan như lỗi đã phát hiện ở câu 1.

    Args:
        answer_text: câu trả lời LLM đã sinh.
        ranked: list context đã rerank (thứ tự điểm cao -> thấp).
        corpus_lookup: dict (article_num, doc_number) -> chunk, build 1 lần bằng
            build_corpus_lookup(). Nếu None, bỏ qua tầng fallback (b) — chỉ match
            trong pool (giống hành vi gốc nhưng vẫn có pairing đúng + bỏ fallback mù).
        pool_k: số ứng viên rerank đầu tiên dùng để match tầng (a).
        max_out: số lượng tối đa context trả về.
    """
    pool = ranked[:pool_k]
    pairs = parse_citation_pairs(answer_text)

    if not pairs:
        # Answer không cite được Điều nào hợp lệ -> fallback top-N rerank (giống bản
        # gốc: N = Settings.RELEVANT_ARTICLES_MAX, thường = 2). Không dùng top-1 vì
        # F2 phạt Recall nặng gấp 4 lần Precision — mất 1 Điều đúng tệ hơn nhiều so
        # với có thêm 1 Điều thừa. Reranker BAAI/bge đã khá tốt nên top-2 rerank
        # hầu hết là đúng ở các câu LLM không cite được gì (hiếm, thường do câu
        # quá ngắn hoặc LLM sinh answer lan man không trích dẫn cụ thể).
        logger.warning("[answer_intersect_v2] Answer không cite Điều nào hợp lệ -> fallback top-N rerank.")
        n = max(1, getattr(Settings, "RELEVANT_ARTICLES_MAX", 2))
        return pool[:n] if pool else []

    # Index pool theo (article_num, doc_number) để match O(1)
    pool_index: Dict[Tuple[str, str], Dict] = {}
    for c in pool:
        md = c.get("metadata", {}) or {}
        anum = _article_num(md.get("article_id", ""))
        dnum = (md.get("doc_number", "") or "").strip()
        if anum and dnum:
            pool_index.setdefault((anum, dnum), c)

    kept: List[Dict] = []
    kept_keys: Set[Tuple[str, str]] = set()
    unresolved_pairs: List[Tuple[str, Optional[str]]] = []

    for art_num, doc_num in pairs:
        candidate = None

        if doc_num is not None:
            # Cặp đầy đủ (Điều X, văn bản Y) -> match chính xác
            candidate = pool_index.get((art_num, doc_num))
            if candidate is None and corpus_lookup is not None:
                candidate = corpus_lookup.get((art_num, doc_num))
        else:
            # answer chỉ nói "Điều X" trần, không kèm số hiệu -> thử match DUY NHẤT
            # nếu trong pool chỉ có đúng 1 văn bản có Điều X đó (tránh nhầm khi có
            # nhiều văn bản cùng đánh số "Điều X").
            same_article_candidates = [
                c for (a, _d), c in pool_index.items() if a == art_num
            ]
            if len(same_article_candidates) == 1:
                candidate = same_article_candidates[0]

        if candidate is not None:
            md = candidate.get("metadata", {}) or {}
            key = (_article_num(md.get("article_id", "")), (md.get("doc_number", "") or "").strip())
            if key not in kept_keys:
                kept_keys.add(key)
                kept.append(candidate)
        else:
            unresolved_pairs.append((art_num, doc_num))

    if unresolved_pairs:
        logger.info(
            f"[answer_intersect_v2] {len(unresolved_pairs)} cặp cite không tra được "
            f"(không có trong pool/corpus, có thể LLM cite sai hoặc bịa): {unresolved_pairs}"
        )

    if not kept:
        # Mọi cặp cite đều không tra được trong pool lẫn corpus (rất có thể LLM bịa
        # toàn bộ số hiệu hoặc số Điều) -> fallback top-N rerank (N = RELEVANT_ARTICLES_MAX).
        # Lý do dùng top-N chứ không top-1: F2 phạt Recall nặng gấp 4 lần Precision,
        # mất thêm 1 Điều đúng tệ hơn nhiều so với có 1 Điều thừa (xem phân tích số
        # liệu: pipeline cũ fallback top-2 rerank -> Recall 0.6907, v2 fallback top-1
        # -> Recall 0.6707, chênh -0.02 chỉ vì đổi top-2 thành top-1 ở đây).
        logger.warning("[answer_intersect_v2] Không cặp cite nào hợp lệ -> fallback top-N rerank.")
        n = max(1, getattr(Settings, "RELEVANT_ARTICLES_MAX", 2))
        return pool[:n] if pool else []

    return kept[:max_out]