"""
src/post_processor.py
Giai đoạn 6: Hậu Xử Lý & Đóng Gói Bài Nộp (Post-Processing)

1. Regex Extraction: Bóc tách "Điều X", "Khoản Y", tên văn bản từ câu trả lời
2. Hard-Mapping: Tra cứu law_manifest.json để điền chuỗi chuẩn
3. Data Validation: Kiểm tra toàn bộ 2000 bản ghi
4. Submission Packaging: Xuất file results.json + zip
"""

import re
import os
import json
import zipfile
import logging
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime

from config.settings import Settings

logger = logging.getLogger(__name__)

# =========================================================
# Patterns trích xuất thông tin pháp lý từ text
# =========================================================

# Tham chiếu điều, khoản, điểm
ARTICLE_FULL_PATTERN = re.compile(
    r'(?:Căn\s+cứ\s+|Theo\s+quy\s+định\s+tại\s+|[Tt]heo\s+|[Tt]ại\s+)?'
    r'([Đđ]iều\s+\d+[a-zA-Z]?)'
    r'(?:,?\s*[Kk]hoản\s+(\d+))?'
    r'(?:,?\s*[Đđ]iểm\s+([a-zA-Z]))?',
    re.IGNORECASE
)

# Số hiệu văn bản pháp luật Việt Nam
DOC_NUMBER_PATTERN = re.compile(
    r'(\d{1,3}/\d{4}/(?:QH|NĐ|TT|QĐ|NQ|CT|TW|UBND|BTC|BGDĐT|BYT|BCA|[A-ZĐ0-9/-]+))',
    re.IGNORECASE
)

# Loại văn bản
DOC_TYPE_KEYWORDS = {
    "luật": "Luật",
    "nghị định": "Nghị định",
    "thông tư": "Thông tư",
    "quyết định": "Quyết định",
    "nghị quyết": "Nghị quyết",
    "chỉ thị": "Chỉ thị",
    "thông tư liên tịch": "Thông tư liên tịch",
    "pháp lệnh": "Pháp lệnh",
    "hiến pháp": "Hiến pháp",
}


class PostProcessor:
    """
    Xử lý hậu kỳ câu trả lời: Trích xuất → Hard-Map → Đóng gói JSON.
    """

    def __init__(self, law_manifest: Optional[Dict] = None):
        """
        Args:
            law_manifest: Dict ánh xạ số hiệu → metadata chuẩn.
                         Cấu trúc mẫu:
                         {
                           "80/2021/NĐ-CP": {
                             "loai_van_ban": "Nghị định",
                             "so_hieu": "80/2021/NĐ-CP",
                             "trich_yeu": "Quy định chi tiết...",
                             "ten_day_du": "Nghị định 80/2021/NĐ-CP ngày..."
                           }
                         }
        """
        self.law_manifest = law_manifest or {}
        logger.info(f"✅ PostProcessor init với {len(self.law_manifest)} entries trong manifest.")

    @classmethod
    def from_manifest_file(cls, path: str = Settings.LAW_MANIFEST_PATH) -> "PostProcessor":
        try:
            with open(path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            return cls(law_manifest=manifest)
        except FileNotFoundError:
            logger.warning(f"Không tìm thấy law_manifest.json tại {path}")
            return cls(law_manifest={})

    # =========================================================
    # BƯỚC 1: REGEX EXTRACTION
    # =========================================================
    def extract_legal_references(self, answer: str) -> Tuple[List[str], List[str]]:
        """
        Bóc tách toàn bộ tham chiếu pháp lý từ câu trả lời.

        Returns:
            Tuple(doc_numbers, article_refs)
            - doc_numbers: ["80/2021/NĐ-CP", ...]
            - article_refs: ["Điều 5", "Điều 5, Khoản 2", ...]
        """
        # Trích số hiệu văn bản
        doc_numbers = list(dict.fromkeys(DOC_NUMBER_PATTERN.findall(answer)))

        # Trích điều/khoản/điểm
        raw_articles = ARTICLE_FULL_PATTERN.findall(answer)
        article_refs = []
        for match in raw_articles:
            dieu, khoan, diem = match
            ref = dieu.strip()
            if khoan:
                ref += f", Khoản {khoan}"
            if diem:
                ref += f", Điểm {diem.lower()}"
            if ref and ref not in article_refs:
                article_refs.append(ref)

        return doc_numbers, article_refs

    # =========================================================
    # BƯỚC 2: HARD-MAPPING → relevant_docs & relevant_articles
    # =========================================================
    def _build_canonical_doc_string(self, doc_number: str) -> Optional[str]:
        """
        Tra manifest để lấy chuỗi chuẩn theo format BTC:
        "[Loại văn bản] [Số hiệu] [Trích yếu]"
        """
        entry = self.law_manifest.get(doc_number)

        if entry is None:
            # Tìm fuzzy trong manifest
            for key, val in self.law_manifest.items():
                if doc_number in key or key in doc_number:
                    entry = val
                    break

        if entry is None:
            # Fallback: Sinh tên từ số hiệu
            return self._infer_doc_type_from_number(doc_number)

        if isinstance(entry, dict):
            loai = entry.get("loai_van_ban", "")
            so_hieu = entry.get("so_hieu", doc_number)
            trich_yeu = entry.get("trich_yeu", "")
            if loai and trich_yeu:
                return f"{loai} {so_hieu} {trich_yeu}"
            elif entry.get("ten_day_du"):
                return entry["ten_day_du"]
        elif isinstance(entry, str):
            return entry

        return None

    def _infer_doc_type_from_number(self, doc_number: str) -> str:
        """Suy luận loại văn bản từ ký hiệu số hiệu."""
        upper = doc_number.upper()
        if "QH" in upper:
            return f"Luật số {doc_number}"
        elif "NĐ-CP" in upper or "ND-CP" in upper:
            return f"Nghị định {doc_number}"
        elif "TT-BTC" in upper or "TT-" in upper:
            return f"Thông tư {doc_number}"
        elif "QĐ" in upper:
            return f"Quyết định {doc_number}"
        else:
            return f"Văn bản {doc_number}"

    def build_relevant_docs(self, doc_numbers: List[str]) -> List[str]:
        """Xây dựng trường relevant_docs theo chuẩn BTC."""
        relevant_docs = []
        seen: Set[str] = set()

        for doc_num in doc_numbers:
            canonical = self._build_canonical_doc_string(doc_num)
            if canonical and canonical not in seen:
                relevant_docs.append(canonical)
                seen.add(canonical)

        return relevant_docs

    def build_relevant_articles(
        self,
        article_refs: List[str],
        doc_numbers: List[str]
    ) -> List[str]:
        """
        Xây dựng trường relevant_articles theo chuẩn BTC.
        Format: "[Điều X, Khoản Y] [Tên văn bản số hiệu]"
        """
        relevant_articles = []
        seen: Set[str] = set()

        # Lấy tên văn bản đầu tiên làm ngữ cảnh
        primary_doc = ""
        if doc_numbers:
            primary_doc = self._build_canonical_doc_string(doc_numbers[0]) or doc_numbers[0]

        for art_ref in article_refs:
            if primary_doc:
                full_ref = f"{art_ref} {primary_doc}"
            else:
                full_ref = art_ref

            if full_ref not in seen:
                relevant_articles.append(full_ref)
                seen.add(full_ref)

        return relevant_articles

    # =========================================================
    # PROCESS SINGLE ITEM
    # =========================================================
    def process_single(
        self,
        item_id: str,
        query: str,
        answer: str,
        context_docs: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Xử lý hậu kỳ một bản ghi đơn lẻ.

        Returns:
            Dict đúng chuẩn format BTC:
            {
                "id": str,
                "question": str,
                "answer": str,
                "relevant_docs": [...],
                "relevant_articles": [...]
            }
        """
        doc_numbers, article_refs = self.extract_legal_references(answer)

        # Bổ sung doc_numbers từ context metadata (nếu LLM quên trích dẫn số hiệu)
        if context_docs:
            for doc in context_docs:
                meta = doc.get("metadata", {})
                dn = meta.get("doc_number", "")
                if dn and dn not in doc_numbers:
                    # Chỉ thêm nếu số hiệu xuất hiện trong câu trả lời
                    if dn in answer or dn.split("/")[0] in answer:
                        doc_numbers.append(dn)

        relevant_docs = self.build_relevant_docs(doc_numbers)
        relevant_articles = self.build_relevant_articles(article_refs, doc_numbers)

        return {
            "id": item_id,
            "question": query,
            "answer": answer,
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
        }

    # =========================================================
    # BƯỚC 3: DATA VALIDATION
    # =========================================================
    def validate_results(self, results: List[Dict]) -> Dict:
        """
        Kiểm tra tổng thể file kết quả trước khi nộp.

        Checks:
        - Đủ 2000 bản ghi
        - Không trùng lặp ID
        - Không có trường null
        - Cú pháp JSON hợp lệ
        - Tất cả answer có tham chiếu pháp lý
        """
        report = {
            "total_records": len(results),
            "errors": [],
            "warnings": [],
            "stats": {
                "no_relevant_docs": 0,
                "no_relevant_articles": 0,
                "no_answer": 0,
                "short_answers": 0,  # Câu trả lời < 50 ký tự
            }
        }

        seen_ids: Set[str] = set()

        for i, item in enumerate(results):
            item_id = item.get("id", f"item_{i}")

            # Kiểm tra trùng ID
            if item_id in seen_ids:
                report["errors"].append(f"Trùng lặp ID: {item_id}")
            seen_ids.add(str(item_id))

            # Kiểm tra null fields
            required_fields = ["id", "question", "answer", "relevant_docs", "relevant_articles"]
            for field in required_fields:
                if item.get(field) is None:
                    report["errors"].append(f"[{item_id}] Trường '{field}' là null")

            # Kiểm tra answer rỗng
            answer = item.get("answer", "")
            if not answer or len(answer) < 10:
                report["stats"]["no_answer"] += 1
                report["errors"].append(f"[{item_id}] Answer rỗng hoặc quá ngắn")
            elif len(answer) < 50:
                report["stats"]["short_answers"] += 1
                report["warnings"].append(f"[{item_id}] Answer ngắn ({len(answer)} chars)")

            # Kiểm tra relevant_docs
            if not item.get("relevant_docs"):
                report["stats"]["no_relevant_docs"] += 1
                report["warnings"].append(f"[{item_id}] relevant_docs rỗng")

            # Kiểm tra relevant_articles
            if not item.get("relevant_articles"):
                report["stats"]["no_relevant_articles"] += 1
                report["warnings"].append(f"[{item_id}] relevant_articles rỗng")

        # Kiểm tra số lượng
        if len(results) != 2000:
            report["errors"].append(
                f"Số bản ghi không đúng: có {len(results)}, cần 2000"
            )

        report["is_valid"] = len(report["errors"]) == 0
        return report

    # =========================================================
    # BƯỚC 4: SUBMISSION PACKAGING
    # =========================================================
    def package_submission(
        self,
        results: List[Dict],
        output_dir: str = Settings.OUTPUT_DIR
    ) -> Tuple[str, str]:
        """
        Xuất results.json và đóng gói thành submission.zip.

        Returns:
            Tuple(json_path, zip_path)
        """
        os.makedirs(output_dir, exist_ok=True)

        # Xuất results.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, "results.json")
        zip_path = os.path.join(output_dir, f"submission_{timestamp}.zip")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Đã xuất results.json: {json_path}")

        # Đóng gói zip (results.json ở thư mục gốc của zip)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname="results.json")
        logger.info(f"📦 Đã đóng gói: {zip_path}")

        return json_path, zip_path
