# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng từ 0.0 lên 1.5, câu trả lời có xu hướng đa dạng và sáng tạo hơn, cách diễn đạt và lựa chọn sự thật cũng thay đổi nhiều hơn. Ở temperature thấp, model thường trả lời ổn định và tập trung vào một thông tin phổ biến; ở temperature cao, phản hồi có nhiều cách diễn đạt và khả năng xuất hiện những ý tưởng ít phổ biến hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ chọn temperature khoảng 0.2–0.3 cho chatbot hỗ trợ khách hàng. Mức temperature thấp giúp câu trả lời ổn định, nhất quán và ít tạo ra thông tin ko cần thiết hoặc khó kiểm soát. Với chatbot chăm sóc khách hàng, độ chính xác và tính nhất quán quan trọng hơn sự sáng tạo.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với workload 10.000 người dùng/ngày × 3 lượt gọi × 350 token đầu ra, tổng lượng output khoảng 10,5 triệu token/ngày. Theo bảng giá dùng trong lab, GPT-4o có giá output cao hơn GPT-4o-mini khoảng 25 lần, nên chi phí output của GPT-4o cũng xấp xỉ 25 lần mini khi cùng lượng token. GPT-4o xứng đáng dùng cho các tác vụ cần chất lượng suy luận và câu trả lời phức tạp, còn GPT-4o-mini phù hợp với FAQ, phân loại yêu cầu hoặc các câu hỏi đơn giản có số lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với system prompt dành cho giáo viên tiểu học, model có xu hướng dùng từ vựng đơn giản, câu ngắn và các ví dụ gần gũi với trẻ em để giải thích blockchain. Với system prompt dành cho chuyên gia tài chính, phản hồi thường sử dụng nhiều thuật ngữ kỹ thuật hơn và có thể đề cập đến các khái niệm như sổ cái phân tán, cơ chế đồng thuận hoặc tài sản số. Vì vậy, cùng một user prompt nhưng system prompt có thể thay đổi đáng kể độ dài, từ vựng, cách giải thích và loại ví dụ được sử dụng. System prompt đóng vai trò định hướng persona và cách model thực hiện yêu cầu.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Đoạn văn test: Việt Nam là một quốc gia có nhiều vùng địa lý và văn hóa khác nhau. Từ những thành phố lớn như Hà Nội và Thành phố Hồ Chí Minh đến các vùng núi, đồng bằng và ven biển, mỗi nơi đều có những nét đặc trưng riêng. Người Việt sử dụng nhiều món ăn quen thuộc như phở, bánh mì và bún chả, đồng thời có nhiều lễ hội truyền thống được tổ chức trong năm. Đất nước cũng có đường bờ biển dài với nhiều cảnh quan tự nhiên đẹp. Sự kết hợp giữa lịch sử, văn hóa và thiên nhiên tạo nên một môi trường đa dạng và hấp dẫn đối với du khách. Kết quả: Đoạn văn có 117 từ và được tiktoken mã hóa thành 138 tokens. Theo cách ước lượng words / 0.75, kết quả là 156 tokens, cao hơn khoảng 13,0% so với số token thực tế. Tiếng Việt thường có thể sử dụng nhiều token hơn tiếng Anh vì cách tách token phụ thuộc vào dữ liệu huấn luyện của tokenizer; các từ tiếng Việt có dấu và cách biểu diễn theo âm tiết có thể khiến một từ được chia thành nhiều token.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming đặc biệt quan trọng với các tác vụ mà model cần nhiều thời gian để tạo phản hồi, chẳng hạn như viết nội dung dài, giải thích một vấn đề phức tạp hoặc tạo mã nguồn. Người dùng có thể bắt đầu đọc ngay khi những token đầu tiên xuất hiện nên cảm giác chờ đợi được giảm xuống, dù tổng thời gian model tạo toàn bộ câu trả lời ko nhất thiết giảm. Ngược lại, non-streaming phù hợp với các tác vụ cần nhận một kết quả hoàn chỉnh trước khi xử lý tiếp, chẳng hạn như một API backend cần parse toàn bộ JSON hoặc thực hiện bước xử lý tiếp theo.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp giảm áp lực lên API khi hệ thống đang quá tải bằng cách tăng dần thời gian chờ giữa các lần retry, ví dụ 0,1 giây, 0,2 giây rồi 0,4 giây. So với delay cố định, cách này làm các request lỗi ko tiếp tục dồn dập vào server. Nếu hàng nghìn client cùng retry với delay cố định giống nhau, chúng có thể retry đồng thời và tạo ra một đợt tải mới ngay khi server vừa phục hồi. Trong hệ thống thực tế, người ta thường kết hợp exponential backoff với jitter để tránh hiện tượng các client retry đồng bộ.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona: Trợ lý học tập lập trình và AI.System prompt: Bạn là trợ lý học tập về lập trình và AI. Hãy giải thích các khái niệm kỹ thuật bằng tiếng Việt rõ ràng, ưu tiên ví dụ thực tế và các bước thực hiện cụ thể. Nếu câu hỏi đơn giản, trả lời ngắn gọn; nếu vấn đề phức tạp, hãy chia thành từng phần dễ theo dõi. Ko tự bịa thông tin khi ko chắc chắn. Tôi chọn yêu cầu bằng tiếng Việt để trợ lý phù hợp với người dùng Việt Nam và tránh câu trả lời quá khó hiểu. Tôi cũng yêu cầu chia thành từng phần dễ theo dõi vì các chủ đề lập trình và AI thường có nhiều bước, giúp người học dễ thực hành và kiểm tra từng phần.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
ko có bộ nhớ dài hạn, ko kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Điểm hạn chế lớn nhất của trợ lý hiện tại là history chỉ giữ tối đa 3 lượt hội thoại gần nhất, vì vậy các thông tin được trao đổi từ những lượt cũ hơn có thể bị mất khỏi context. Một cải thiện cụ thể là xây dựng long-term memory bằng cách lưu các thông tin quan trọng của cuộc hội thoại vào cơ sở dữ liệu hoặc vector database. Khi có tin nhắn mới, hệ thống có thể tìm kiếm các thông tin liên quan và thêm chúng vào context cùng với 3 lượt hội thoại gần nhất. Cách này giúp giảm lượng token phải gửi nhưng vẫn duy trì được những thông tin quan trọng trong thời gian dài.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
