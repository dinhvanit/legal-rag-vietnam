import re
import logging
import time
from typing import List, Dict, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

from config.settings import Settings

logger = logging.getLogger(__name__)

DOC_NUMBER_PATTERN = re.compile(r'\d{1,3}/\d{4}/[A-ZĐ0-9/-]+')

# =========================================================
# SYSTEM PROMPT — Tối ưu cho DeepSeek-R1 ép suy nghĩ bằng tiếng Việt
# =========================================================
SYSTEM_PROMPT = """Bạn là một robot trợ lý ảo tư vấn pháp luật chuyên nghiệp cho khối doanh nghiệp vừa và nhỏ (SME) tại Việt Nam.

NHIỆM VỤ: Hãy trả lời câu hỏi pháp lý của người dùng một cách chính xác, ngắn gọn và khách quan.
Nghiêm cấm suy nghĩ (CoT) bằng tiếng Anh. Toàn bộ quá trình suy nghĩ và câu trả lời PHẢI viết bằng tiếng Việt.

🚨 QUY TẮC AN TOÀN PHÁP LÝ TUYỆT ĐỐI KHÔNG ĐƯỢC VI PHẠM (CHỐNG ẢO GIÁC):
1. Bạn CHỈ ĐƯỢC PHÉP trích dẫn các số hiệu Điều, Khoản, Điểm luật xuất hiện MỘT CÁCH TƯỜNG MINH trong phần [NGỮ CẢNH PHÁP LÝ] được cung cấp.
2. TUYỆT ĐỐI KHÔNG tự bịa đặt, suy diễn, hoặc đoán mò số thứ tự Điều/Khoản hoặc số hiệu văn bản.
3. Nếu thông tin trong [NGỮ CẢNH PHÁP LÝ] không chứa câu trả lời, bạn phải ghi rõ: "Dựa trên ngữ cảnh pháp lý được cung cấp, không có quy định cụ thể về vấn đề này."
4. Mọi trích dẫn tại mục 2 bắt buộc phải ghi rõ: "Căn cứ Điều X, Khoản Y của [Tên văn bản hoặc số hiệu]".

ĐỊNH DẠNG TRẢ LỜI BẮT BUỘC (PHẢI GIỮ NGUYÊN TIÊU ĐỀ ##):

## 1. KẾT LUẬN
[Trả lời thẳng vào vấn đề, ngắn gọn từ 1 - 3 câu]

## 2. CĂN CỨ PHÁP LÝ
[Chỉ liệt kê các Điều, Khoản thực tế có trong ngữ cảnh. Định dạng: Căn cứ Điều X, Khoản Y của [Số hiệu văn bản]]

## 3. PHÂN TÍCH CHI TIẾT
[Phân tích làm rõ câu hỏi dựa trên các căn cứ đã nêu ở mục 2]

## 4. LƯU Ý
Nội dung tư vấn trên được tổng hợp từ các quy định pháp luật hiện hành và chỉ mang tính tham khảo. Để được tư vấn chính xác và đầy đủ cho trường hợp cụ thể, doanh nghiệp nên tham khảo ý kiến của luật sư hoặc cơ quan nhà nước có thẩm quyền."""

def build_user_prompt(query: str, contexts: List[Dict]) -> str:
    context_parts = []
    total_chars = 0

    for i, doc in enumerate(contexts, start=1):
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})

        doc_number = metadata.get("doc_number", "")
        doc_name = metadata.get("doc_name", "")
        article = metadata.get("article", "")

        header = f"[Văn bản {i}]"
        if doc_name:
            header += f" {doc_name}"
        if doc_number:
            header += f" - Số hiệu: {doc_number}"
        if article:
            header += f" - {article}"

        chunk = f"{header}\n{text}"

        if total_chars + len(chunk) > Settings.MAX_CONTEXT_CHARS:
            remaining = Settings.MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:
                chunk = chunk[:remaining] + "...[cắt bớt]"
                context_parts.append(chunk)
            break

        context_parts.append(chunk)
        total_chars += len(chunk)

    context_text = "\n\n" + "\n\n---\n\n".join(context_parts) + "\n"

    prompt = f"""[NGỮ CẢNH PHÁP LÝ]
{context_text}
[CÂU HỎI]
{query}

[YÊU CẦU]
Hãy suy nghĩ bằng tiếng Việt và trả lời câu hỏi trên theo đúng cấu trúc 4 phần đã quy định, chỉ dựa vào nội dung trong [NGỮ CẢNH PHÁP LÝ]."""

    return prompt


class AnswerGenerator:
    def __init__(self):
        logger.info(f"[LLM] Đang load model: {Settings.LLM_MODEL_NAME}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            Settings.LLM_MODEL_NAME,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            Settings.LLM_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model.eval()

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        logger.info("✅ LLM load thành công!")

    def generate(
        self,
        query: str,
        contexts: List[Dict],
        temperature: float = Settings.LLM_TEMPERATURE
    ) -> str:
        user_prompt = build_user_prompt(query, contexts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            formatted_prompt = f"{SYSTEM_PROMPT}\n\nNgười dùng: {user_prompt}\n\nTrợ lý:"

        start = time.time()
        
        # CHÚ Ý: Đổi do_sample=True nếu dùng temperature, tăng tối thiểu max_new_tokens lên 2048 hoặc 3072 cho R1
        max_tokens = max(2048, getattr(Settings, "LLM_MAX_NEW_TOKENS", 2048))
        
        output = self.pipe(
            formatted_prompt,
            max_new_tokens=max_tokens, 
            temperature=0.6 if temperature == 0 else temperature, # DeepSeek-R1 khuyến khích khuyên dùng 0.6 TPC
            top_p=Settings.LLM_TOP_P,
            repetition_penalty=Settings.LLM_REPETITION_PENALTY,
            do_sample=True, # Bật lấy mẫu để kiểm soát sáng tạo và hội thoại tự nhiên hơn
        )
        elapsed = time.time() - start
        logger.debug(f"[LLM] Sinh xong trong {elapsed:.1f}s")

        generated_text = output[0]["generated_text"]

        # 1. Bóc tách phần sinh phản hồi của Trợ lý
        if formatted_prompt in generated_text:
            answer = generated_text[len(formatted_prompt):].strip()
        else:
            answer = generated_text.split("Trợ lý:")[-1].strip()

        # 2. Xử lý triệt để thẻ suy nghĩ <think>...</think>
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        answer = re.sub(r'<think>.*', '', answer, flags=re.DOTALL).strip() # Backup nếu bị cắt cụt ngay trong thẻ think

        # 3. Clean up toàn bộ tàn dư tiếng Anh do hiện tượng rò rỉ CoT ngoài thẻ (nếu có)
        lines = answer.split('\n')
        filtered_lines = []
        is_legal_start = False
        
        for line in lines:
            # Nhận diện điểm bắt đầu cấu trúc thực tế bằng Tiếng Việt
            if "## 1." in line or "KẾT LUẬN" in line:
                is_legal_start = True
            
            if not is_legal_start:
                # Nếu chưa tới phần kết luận mà dính các từ khóa tiếng Anh của CoT -> bỏ qua
                if re.match(r'^(Okay|So|First|Let me|I need|Looking at|Moving on|Putting it|We have|According to)', line.strip(), re.I):
                    continue
                # Bỏ qua các dòng trống hoặc dòng text tiếng Anh linh tinh trước khi vào cấu trúc chính
                if any(en_word in line.lower() for en_word in ['concept', 'document', 'paragraph', 'the user is asking']):
                    continue
            
            filtered_lines.append(line)
            
        answer = '\n'.join(filtered_lines).strip()
        return answer