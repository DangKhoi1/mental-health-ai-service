import json
import random
import traceback

from fastapi import FastAPI
from openai import OpenAI
import uvicorn

from .schemas import ChatRequest, ChatResponse, SentimentResult
from .services.sentiment import analyze_sentiment
from .services.response_generator import generate_response
from .core.config import OPENAI_API_KEY, MODEL


app = FastAPI(
    title="Mental Health Chatbot AI",
    description="Chatbot hỗ trợ tinh thần dựa trên phân tích cảm xúc và OpenAI",
    version="2.0",
)

_client = None


def get_openai_client() -> OpenAI:
    """Lazy initialization of OpenAI client."""
    global _client
    if _client is None:
        try:
            _client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception as e:
            print(f"WARNING: Failed to initialize OpenAI client: {e}")
            raise
    return _client


def get_system_prompt(level: str = None) -> str:
    """Generate system prompt based on severity level."""
    base_prompt = """Bạn là một chuyên gia tư vấn sức khỏe tâm thần chuyên nghiệp, thân thiện và đầy empathy. 
Nhiệm vụ của bạn là:
1. Lắng nghe và thấu hiểu cảm xúc của người dùng
2. Đưa ra lời khuyên dựa trên bằng chứng y khoa và tâm lý học
3. Trả về phản hồi bằng tiếng Việt, tự nhiên và ấm áp
4. QUAN TRỌNG: Luôn trả về JSON format với cấu trúc:
{
  "bot_reply": "Lời phản hồi của bạn ở đây",
  "recommendations": [
    {"category": "TÊN_CATEGORY", "content": "Nội dung khuyến nghị cụ thể"}
  ]
}

Categories hợp lệ: PROFESSIONAL, SLEEP, MEDITATION, BREATHING, EXERCISE, SOCIAL, JOURNALING, RELAXATION, NUTRITION
"""

    severity_guidance = {
        "SEVERE": "\n\nNgười dùng đang ở mức độ NGHIÊM TRỌNG. Ưu tiên khuyến nghị tìm kiếm sự giúp đỡ chuyên nghiệp ngay lập tức (PROFESSIONAL), kết hợp hỗ trợ xã hội (SOCIAL) và các kỹ thuật cơ bản (BREATHING, SLEEP).",
        "MODERATELY_SEVERE": "\n\nNgười dùng ở mức độ KHÁ NGHIÊM TRỌNG. Khuyến khích tư vấn chuyên gia (PROFESSIONAL), kết hợp với các hoạt động tự chăm sóc như thiền (MEDITATION), thể dục nhẹ (EXERCISE), và viết nhật ký (JOURNALING).",
        "MODERATE": "\n\nNgười dùng ở mức độ TRUNG BÌNH. Tập trung vào tự chăm sóc tích cực: thể dục (EXERCISE), thiền (MEDITATION), kết nối xã hội (SOCIAL), và dinh dưỡng (NUTRITION).",
        "MILD": "\n\nNgười dùng ở mức độ NHẸ. Khuyến nghị phòng ngừa: thư giãn (RELAXATION), giấc ngủ chất lượng (SLEEP), hoạt động xã hội (SOCIAL), và nhật ký biết ơn (JOURNALING).",
        "MINIMAL": "\n\nNgười dùng ở trạng thái TỐT. Khuyến nghị duy trì wellness: tiếp tục vận động (EXERCISE), chia sẻ tích cực (SOCIAL), thử thách mới (RELAXATION), và thiền (MEDITATION).",
    }

    if level:
        level_upper = level.upper()
        for key in severity_guidance:
            if key in level_upper:
                return base_prompt + severity_guidance[key]

    return base_prompt


def get_recommendations_system_prompt() -> str:
    return """Bạn là chuyên gia sức khỏe tâm thần và phong cách sống sáng tạo. Dựa trên thống kê hoạt động của người dùng trong 7 ngày qua (tâm trạng, giấc ngủ, nhật ký), hãy đưa ra ĐÚNG 3 lời khuyên NHỎ, CỤ THỂ và ĐỘC ĐÁO để cải thiện sức khỏe tinh thần. 

TRÁNH các lời khuyên chung chung kiểu "ngủ đủ giấc", "tập thể dục", "hít thở sâu" lặp đi lặp lại.
HÃY TƯỞNG TƯỢNG VÀ SÁNG TẠO các hoạt động nhỏ nhắn, dể thương nhưng hiệu quả: ví dụ "Thử vẽ một cái cây bằng tay trái", "Mua một bông hoa tặng bản thân", "Nghe podcast về vũ trụ trước khi ngủ", "Nấu một món chưa từng nấu", "Xóa 50 ảnh cũ trong điện thoại để cảm thấy nhẹ nhõm hơn"...

LƯU Ý TỐI QUAN TRỌNG: TOÀN BỘ NỘI DUNG Văn bản gửi cho người dùng (bot_reply, title, content) PHẢI ĐƯỢC VIẾT BẰNG TIẾNG VIỆT 100%. Tuyệt đối KHÔNG ĐƯỢC chèn bất kỳ từ tiếng Anh nào vào câu (ngoại trừ các Category bắt buộc quy định bên dưới).

QUAN TRỌNG: Bạn PHẢI trả về JSON hợp lệ theo cấu trúc sau (không có text ngoài JSON):
{
  "bot_reply": "Tổng quan ngắn 1 câu về tình trạng người dùng",
  "recommendations": [
    {"title": "Tiêu đề sáng tạo (Tối đa 5 từ)", "category": "SLEEP|MEDITATION|EXERCISE|JOURNALING|SOCIAL|BREATHING|RELAXATION|NUTRITION|PROFESSIONAL", "content": "Nội dung hành động cụ thể. Giọng văn khích lệ, gần gũi."},
    {"title": "Tiêu đề sáng tạo", "category": "...", "content": "..."},
    {"title": "Tiêu đề sáng tạo", "category": "...", "content": "..."}
  ]
}"""


def get_chat_system_prompt_with_resources(base_prompt: str, resources: list) -> str:
    if not resources:
        return base_prompt
    resource_lines = "\n".join(
        f'- {r.get("title","")} ({r.get("description","")[:50]}...)'
        for r in resources[:5]
    )
    return (
        base_prompt
        + f"\n\nDanh sách các bài tập/tài nguyên chữa lành sẵn có trên hệ thống:\n{resource_lines}\n\nNẾU BẠN GỢI Ý TÀI NGUYÊN, BẠN BẮT BUỘC PHẢI GHI RÕ TÊN BÀI TẬP ĐÓ RA TRONG CÂU TRẢ LỜI CỦA MÌNH (Ví dụ: Bạn có thể thử bài tập 'Thiền Chánh Niệm Cơ Bản' trong Thư viện nhé)."
    )


def parse_openai_response(content: str) -> dict:
    """Parse OpenAI response to extract bot_reply and recommendations."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"bot_reply": content, "recommendations": []}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        sentiment = analyze_sentiment(request.message)

        # ── Branch 1: Dashboard recommendations ────────────────────────────
        action = (request.context or {}).get("action", "")
        if action == "generate_dashboard_recommendations":
            stats = (request.context or {}).get("stats", {})
            stats_text = json.dumps(stats, ensure_ascii=False, indent=2)
            user_prompt = (
                f"Thống kê 7 ngày qua của người dùng:\n{stats_text}\n\n"
                "Hãy phân tích và trả về đúng 3 lời khuyên cải thiện sức khỏe tinh thần theo format JSON yêu cầu. Nhớ BẤT NGỜ và KHÁC BIỆT so với những lần trước."
            )
            try:
                client = get_openai_client()
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": get_recommendations_system_prompt()},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.9,
                    timeout=30.0,
                )
                ai_response = parse_openai_response(response.choices[0].message.content)
                bot_reply = ai_response.get("bot_reply", "Đây là các gợi ý cải thiện sức khỏe tinh thần của bạn.")
                recommendations = ai_response.get("recommendations", [])
            except Exception as e:
                print(f"WARNING: OpenAI error (dashboard recommendations): {e}\n{traceback.format_exc()}")
                bot_reply = "Dựa trên hoạt động gần đây, đây là một số gợi ý cho bạn."
                recommendations = [
                    {"title": "Cải thiện giấc ngủ", "category": "SLEEP", "content": "Hãy cố gắng đi ngủ và thức dậy vào cùng một giờ mỗi ngày để điều chỉnh đồng hồ sinh học."},
                    {"title": "Thiền và hít thở", "category": "BREATHING", "content": "Dành 5–10 phút mỗi sáng để luyện tập hít thở sâu giúp giảm căng thẳng hiệu quả."},
                    {"title": "Viết nhật ký cảm xúc", "category": "JOURNALING", "content": "Ghi lại 3 điều bạn biết ơn mỗi tối giúp duy trì tư duy tích cực."},
                ]
            return ChatResponse(
                sentiment=SentimentResult(score=sentiment["score"], mood=sentiment["mood"]),
                bot_reply=bot_reply,
                recommendations=recommendations,
            )

        # ── Branch 2: Assessment & general chat ────────────────────────────
        has_valid_context = bool(request.context and request.context.get("title"))
        is_assessment = has_valid_context or "kết quả đánh giá" in request.message.lower()

        if is_assessment:
            level = request.context.get("level", "MODERATE") if request.context else "MODERATE"

            if request.context:
                assessment_info = f"""
Thông tin đánh giá của người dùng:
- Bài đánh giá: {request.context.get('title', 'N/A')}
- Mức độ: {request.context.get('level', 'N/A')}
- Điểm số: {request.context.get('totalScore', 'N/A')}/{request.context.get('maxScore', 'N/A')}
- Thông điệp: {request.context.get('message', 'N/A')}

Người dùng nói: {request.message}

Hãy phân tích kết quả và đưa ra 4 khuyến nghị cụ thể, phù hợp với mức độ nghiêm trọng.
"""
            else:
                assessment_info = f"Người dùng nói: {request.message}\n\nHãy đưa ra lời khuyên và các khuyến nghị phù hợp."

            try:
                client = get_openai_client()
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": get_system_prompt(level)},
                        {"role": "user", "content": assessment_info},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    timeout=60.0,
                )
                ai_response = parse_openai_response(response.choices[0].message.content)
                bot_reply = ai_response.get("bot_reply", "Xin lỗi, tôi gặp sự cố kỹ thuật.")
                recommendations = ai_response.get("recommendations", [])

            except Exception as e:
                print(f"WARNING: OpenAI API Error (Assessment): {type(e).__name__}: {e}")
                traceback.print_exc()

                level_upper = level.upper() if isinstance(level, str) else "MODERATE"

                if "MINIMAL" in level_upper or "TỐI THIỂU" in level_upper:
                    bot_reply = """Tuyệt vời! Sức khỏe tinh thần của bạn đang ở trạng thái rất tốt (Mức độ: Tối thiểu).
                    
**Khuyến nghị duy trì:**
- Trân trọng những khoảnh khắc tích cực mỗi ngày.
- Đảm bảo duy trì thói quen ngủ nghỉ hợp lý và vận động nhẹ nhàng.
- Theo dõi định kỳ để hiểu rõ bản thân hơn.

Bạn có muốn mình chia sẻ thêm vài mẹo nhỏ để giữ vững tinh thần thoải mái này không?"""
                elif "MILD" in level_upper or "NHẸ" in level_upper:
                    bot_reply = """Dựa trên kết quả đánh giá (Mức độ: Nhẹ), có vẻ bạn đang gặp chút căng thẳng nhưng vẫn trong tầm kiểm soát.

**Khuyến nghị trong tuần tới:**
- Dành ra 15 phút mỗi tối để làm việc mình thích (đọc sách, nghe nhạc).
- Thử tập các bài hít thở sâu 4-4-6 khi cảm thấy áp lực.
- Tránh làm việc thiết bị điện tử 1 tiếng trước khi ngủ.

Bạn có cần mình hướng dẫn cụ thể bài tập hít thở thư giãn không?"""
                elif "MODERATE" in level_upper or "TRUNG BÌNH" in level_upper:
                    bot_reply = """Dựa trên kết quả (Mức độ: Trung Bình), hiện tại bạn đang chịu khá nhiều áp lực định kỳ.

**Lộ trình 2 tuần tới:**
*Tuần 1: Ổn định*
- Ghi chép nhật ký cảm xúc để tìm ra nguyên nhân khiến bạn mệt mỏi nhất.
- Yêu cầu sự giúp đỡ từ bạn bè hoặc đồng nghiệp trong các công việc quá tải.

*Tuần 2: Phục hồi*
- Thực hành chia sẻ cảm xúc với ít nhất 1 người đáng tin cậy.
- Đi dạo hoặc vận động ngoài trời 30 phút mỗi ngày.

Mình luôn ở đây lắng nghe, bạn có muốn kể chi tiết điều gì đang khiến bạn trăn trở không?"""
                else:
                    bot_reply = """Kết quả đánh giá cho thấy bạn đang trải qua khoảng thời gian vô cùng khó khăn (Mức độ: Khá nghiêm trọng/Nghiêm trọng). Mình rất tiếc khi nghe điều này.

**Hành động khẩn cấp:**
- Đừng cố gắng vượt qua một mình. Hãy tìm kiếm sự hỗ trợ chuyên môn từ bác sĩ tâm lý hoặc tổng đài tư vấn ngay khi có thể.
- Tạm gác lại mọi áp lực không thiết yếu để ưu tiên nghỉ ngơi tuyệt đối.
- Ở cạnh người thân hoặc bạn bè mà bạn tin tưởng nhất lúc này.

Bạn rất dũng cảm khi đối diện với cảm xúc của mình. Bạn có muốn mình cung cấp một số thông tin liên hệ chuyên gia hoặc tổ chức hỗ trợ không?"""
                recommendations = []

        else:
            # ── Branch 3: General chat ──────────────────────────────────────
            healing_resources = (request.context or {}).get("healingLibrary", [])
            personal_stats = (request.context or {}).get("personalStats", {})
            user_name = (request.context or {}).get("userName", "Người dùng")

            base_prompt = get_system_prompt()

            if personal_stats:
                mood_info = personal_stats.get("moodStats", {})
                recent_thoughts = personal_stats.get("recentThoughts", [])
                stats_str = f"\nThông tin người dùng ({user_name}):"
                if mood_info:
                    stats_str += f"\n- Cảm xúc hôm nay: {mood_info}"
                if recent_thoughts:
                    stats_str += f"\n- Trăn trở gần đây: {recent_thoughts}"
                base_prompt += f"\n\nBối cảnh cá nhân hóa (Hãy sử dụng tinh tế):\n{stats_str}"

            chat_system_prompt = get_chat_system_prompt_with_resources(base_prompt, healing_resources)

            messages = [{"role": "system", "content": chat_system_prompt}]
            if request.history:
                for msg in request.history:
                    role = msg.get("role")
                    content = msg.get("content")
                    if role in ["user", "assistant"] and content:
                        messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": request.message})

            try:
                client = get_openai_client()
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.8,
                    timeout=30.0,
                )
                ai_response = parse_openai_response(response.choices[0].message.content)
                bot_reply = ai_response.get("bot_reply", generate_response(sentiment["mood"]))
                recommendations = ai_response.get("recommendations", [])

            except Exception as e:
                print(f"WARNING: OpenAI API Error (Chat): {type(e).__name__}: {e}")
                traceback.print_exc()

                msg_lower = request.message.lower()
                if "stress" in msg_lower or "căng thẳng" in msg_lower:
                    bot_reply = random.choice([
                        "Khi cảm thấy stress, bạn hãy thử nhắm mắt lại, hít một hơi thật sâu trong 4 giây, giữ 4 giây, và thở ra từ từ 6 giây nhé. Hoặc đơn giản là đứng lên đi dạo một lát, uống một ngụm nước lọc. Mình luôn ở đây nghe bạn kể.",
                        "Mình hiểu cảm giác này. Những lúc căng thẳng, hãy ưu tiên bản thân mình trước. Tạm rời xa màn hình 10 phút, nghe một bài nhạc không lời hoặc rửa mặt bằng nước mát nhé. Điều gì đang làm bạn bận tâm nhất lúc này?",
                        "Căng thẳng giống như một chiếc lò xo bị ép chặt vậy. Hãy cho phép bản thân bung ra một chút. Mình gợi ý bạn thử đứng lên vươn vai và hít thở sâu 3 lần xem sao nhé.",
                    ])
                elif "mệt" in msg_lower or "chán" in msg_lower:
                    bot_reply = random.choice([
                        "Nghe này, đôi khi cảm thấy mệt mỏi và chán nản là chuyện rất bình thường. Đừng ép bản thân quá. Có lẽ hôm nay cơ thể và tinh thần bạn đang cần nghỉ ngơi. Đi tắm nước ấm, nghe một bản nhạc nhẹ hoặc đi ngủ sớm nhé!",
                        "Nếu hôm nay quá mệt mỏi, bạn hoàn toàn có quyền được 'ngắt kết nối' một lúc. Hãy làm gì đó chỉ để chiều chuộng bản thân thôi, dù là ăn một món ngon hay nằm nhắm mắt thư giãn.",
                        "Bạn đã vất vả rồi. Hãy tự nhủ rằng 'mình đã làm tốt nhất có thể' và tạm gác lại mọi âu lo. Nghỉ ngơi thật tốt, ngày mai sẽ là một ngày mới trọn vẹn hơn.",
                    ])
                elif "vui" in msg_lower or "hạnh phúc" in msg_lower or "tuyệt" in msg_lower:
                    bot_reply = random.choice([
                        "Chà, thật tuyệt vời! Năng lượng tích cực của bạn truyền sang cả mình đấy. Cảm ơn bạn vì đã chia sẻ niềm vui này. Nhớ giữ gìn và phát huy những điều làm bạn mỉm cười hôm nay nhé!",
                        "Nghe bạn nói vui làm mình cũng thấy hạnh phúc lây! Hãy tận hưởng trọn vẹn khoảnh khắc tuyệt vời này nhé. Có điều gì đặc biệt đã giúp hôm nay của bạn trở nên tươi sáng vậy?",
                        "Thật tuyệt khi nghe được những điều tích cực từ bạn. Hãy ghi nhớ cảm giác này, nó sẽ là nguồn năng lượng dự trữ cho bạn trong những ngày tiếp theo!",
                    ])
                else:
                    bot_reply = generate_response(sentiment["mood"])
                recommendations = []

        return ChatResponse(
            sentiment=SentimentResult(score=sentiment["score"], mood=sentiment["mood"]),
            bot_reply=bot_reply,
            recommendations=recommendations,
        )

    except Exception as top_e:
        print(f"CRITICAL ERROR IN /chat: {top_e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    uvicorn.run("app.main:app", reload=True, port=5001)
