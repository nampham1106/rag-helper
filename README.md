### Python package for support RAG 

#### Installation
```pip install rag-helper```

#### Usage
1. Generate synthetic data
Note: only support for Gemini model

```python
from rag_helper.synthetic import SyntheticData

chunks = ["Mang thai và sinh con là thiên chức của người phụ nữ, ai cũng mong muốn con mình sinh ra khỏe mạnh, lành lặn không mắc dị tật. Để hoàn thành được nguyện vọng đó, bên cạnh việc ăn uống, nghỉ ngơi hợp lý thì thăm khám thai định kỳ và thực hiện các chẩn đoán trước sinh là rất cần thiết.. Thông qua các xét nghiệm, siêu âm,.. các bác sĩ cho biết thai nhi trong bụng có phát triển khỏe mạnh không, có mắc dị tật bẩm sinh nào không... Hầu hết chị em phụ nữ mang thai đều được bác sĩ khuyến cáo nên thực hiện sàng lọc, chẩn đoán trước sinh tại các cơ sở uy tín. Vì vậy, để trẻ sinh ra hoàn toàn khỏe mạnh các mẹ bầu cần tuân theo đúng những chỉ dẫn của bác sĩ chuyên khoa, thực hiện sàng lọc trước sinh vào những cột mốc quan trọng.."]

GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
GenData = SyntheticData(GOOGLE_API_KEY)
question_answer = await GenData.generate_synthetic_data(chunks, 1)
```
