"""
score_retrieval.py  (CÔNG CỤ ĐO — không ảnh hưởng pipeline)

Chấm điểm TRUY HỒI của một file results.json so với ground truth, IN macro Precision/Recall/F2
đúng công thức BTC (F2 tính từ macro P & macro R). Dùng để dò siêu tham số OFFLINE trên
ground_truth_50 mà KHÔNG tốn lượt nộp Dashboard.

So khớp: chuẩn hoá về (số hiệu văn bản, số Điều) cho articles; (số hiệu) cho docs — giống cách
hệ thống chấm của BTC rút "Điều X" + mã văn bản.

Chạy:
    python score_retrieval.py --pred output/results.json --gt data/ground_truth_50.json
    python score_retrieval.py --pred output/results.json --gt data/ground_truth_50.json --show-worst 10
"""
import re
import json
import argparse
from typing import Dict, List, Optional, Set, Tuple

ARTICLE_NUM_RE = re.compile(r"[Đđ]iều\s+(\d+)")


def parse_doc(s: str) -> str:
    """'04/2017/QH14|Luật...' -> '04/2017/QH14'"""
    return s.split("|")[0].strip()


def parse_article(s: str) -> Optional[Tuple[str, str]]:
    """'04/2017/QH14|Luật...|Điều 12' -> ('04/2017/QH14', '12')"""
    parts = s.split("|")
    if len(parts) < 2:
        return None
    m = ARTICLE_NUM_RE.search(parts[-1])
    return (parts[0].strip(), m.group(1)) if m else None


def doc_set(item: Dict) -> Set[str]:
    return {parse_doc(d) for d in item.get("relevant_docs", []) if d}


def article_set(item: Dict) -> Set[Tuple[str, str]]:
    out = set()
    for a in item.get("relevant_articles", []):
        p = parse_article(a)
        if p:
            out.add(p)
    return out


def f2(p: float, r: float) -> float:
    return 0.0 if (4 * p + r) == 0 else 5 * p * r / (4 * p + r)


def load_map(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else raw.get("data", raw.get("questions", []))
    return {str(it.get("id", it.get("question_id", ""))): it for it in items}


def main():
    ap = argparse.ArgumentParser(description="Chấm F2 truy hồi local vs ground truth")
    ap.add_argument("--pred", required=True, help="File results.json cần chấm")
    ap.add_argument("--gt", default="data/ground_truth_50.json", help="File ground truth")
    ap.add_argument("--show-worst", type=int, default=0, help="In N câu article F2 thấp nhất")
    args = ap.parse_args()

    pred = load_map(args.pred)
    gt = load_map(args.gt)

    common = [qid for qid in gt if qid in pred]
    if not common:
        print("❌ Không có id chung giữa pred và gt!")
        return

    ap_sum = ar_sum = dp_sum = dr_sum = 0.0
    n = 0
    rows = []
    for qid in common:
        gA, pA = article_set(gt[qid]), article_set(pred[qid])
        gD, pD = doc_set(gt[qid]), doc_set(pred[qid])
        # Bỏ câu GT rỗng (không có nhãn để chấm)
        if not gA and not gD:
            continue
        n += 1
        a_p = len(gA & pA) / len(pA) if pA else 0.0
        a_r = len(gA & pA) / len(gA) if gA else 0.0
        d_p = len(gD & pD) / len(pD) if pD else 0.0
        d_r = len(gD & pD) / len(gD) if gD else 0.0
        ap_sum += a_p; ar_sum += a_r; dp_sum += d_p; dr_sum += d_r
        rows.append((qid, f2(a_p, a_r), sorted(gA - pA), sorted(pA - gA)))

    AP, AR = ap_sum / n, ar_sum / n
    DP, DR = dp_sum / n, dr_sum / n

    print(f"Đã chấm {n} câu (có nhãn) / {len(common)} câu trùng id / pred {len(pred)} câu.\n")
    print(f"{'NHÓM':<10}{'Precision':>11}{'Recall':>9}{'F2':>9}")
    print("-" * 39)
    print(f"{'ARTICLES':<10}{AP:>11.4f}{AR:>9.4f}{f2(AP, AR):>9.4f}")
    print(f"{'DOCS':<10}{DP:>11.4f}{DR:>9.4f}{f2(DP, DR):>9.4f}")

    if args.show_worst:
        print(f"\n--- {args.show_worst} câu ARTICLE F2 thấp nhất (thiếu / thừa) ---")
        for qid, fa, missing, extra in sorted(rows, key=lambda x: x[1])[:args.show_worst]:
            print(f"  Q{qid}: F2={fa:.2f} | THIẾU={missing} | THỪA={extra}")


if __name__ == "__main__":
    main()
