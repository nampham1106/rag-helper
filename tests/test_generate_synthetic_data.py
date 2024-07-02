from rag_helper.synthetic import SyntheticData
from rag_helper.gemini_helper import GeminiHelper

chunks = [ 
"""Nếu mẹ bầu chưa lựa chọn được bác sĩ nào khám và thực hiện chẩn đoán trước sinh giỏi thì có thể tham khảo danh sách trong bài viết dưới đây.. Mang thai và sinh con là thiên chức của người phụ nữ, ai cũng mong muốn con mình sinh ra khỏe mạnh, lành lặn không mắc dị tật. Để hoàn thành được nguyện vọng đó, bên cạnh việc ăn uống, nghỉ ngơi hợp lý thì thăm khám thai định kỳ và thực hiện các chẩn đoán trước sinh là rất cần thiết.. Thông qua các xét nghiệm, siêu âm,.. các bác sĩ cho biết thai nhi trong bụng có phát triển khỏe mạnh không, có mắc dị tật bẩm sinh nào không... Hầu hết chị em phụ nữ mang thai đều được bác sĩ khuyến cáo nên thực hiện sàng lọc, chẩn đoán trước sinh tại các cơ sở uy tín. Vì vậy, để trẻ sinh ra hoàn toàn khỏe mạnh các mẹ bầu cần tuân theo đúng những chỉ dẫn của bác sĩ chuyên khoa, thực hiện sàng lọc trước sinh vào những cột mốc quan trọng.. 8 bác sĩ chẩn đoán trước sinh giỏi tại Hà Nội. Không chỉ quan tâm tới địa chỉ mà việc bác sĩ nào thực hiện chẩn đoán, sàng lọc trước sinh uy tín cũng được các mẹ bầu rất quan tâm. Nếu mẹ bầu chưa lựa chọn được bác sĩ nào khám và thực hiện chẩn đoán trước sinh giỏi thì có thể tham khảo danh sách dưới đây..
1. Phó Giáo sư, Tiến sĩ, Bác sĩ Nguyễn Duy Ánh.
Giám đốc Bệnh viện Phụ sản Hà Nội.
Cố vấn chuyên môn tại Trung tâm chẩn đoán trước sinh và sơ sinh - Bệnh viện Phụ sản Hà Nội.
Chuyên gia về lĩnh vực Sản Phụ khoa - chăm sóc sức khỏe sinh sản. Phó trưởng bộ môn Sản, trường Đại học Y Hà Nội.
Trưởng bộ môn Sản, khoa Y, trường Đại học Quốc gia Hà Nội.
Chuyên gia lĩnh vực Hỗ trợ sinh sản. Chuyên gia lĩnh vực Chẩn đoán trước sinh và Sàng lọc sơ sinh. Giảng viên quốc gia về lĩnh vực chăm sóc sức khỏe sinh sản. Danh hiệu thầy thuốc nhân dân năm 2017. Hiện nay, Phó Giáo sư Ánh có thực hiện khám và siêu âm sàng lọc trước sinh cho phụ nữ mang thai tại phòng khám riêng của bác ở số 21 Vạn Phúc, Liễu Giai, Ba Đình, Hà Nội. Thời gian khám từ 17h – 21h từ thứ 2 - thứ 6, từ 8h – 12h30 thứ 7 và chủ nhật. Điện thoại liên hệ đặt lịch khám 0243 7624 646.. Phó Giáo sư, Bác sĩ Nguyễn Duy Ánh (cầm hoa).
2. Phó Giáo sư, Tiến sĩ, Bác sĩ Trần Danh Cường. Phó Giám đốc Bệnh viện phụ sản Trung Ương.
Giám đốc trung tâm Chẩn đoán trước sinh - Bệnh viện Phụ Sản Trung ương."""
]

async def test():
    gemini = GeminiHelper("AIzaSyC2GxQOE5Nl-SgrAdUvjsnGVTCe0nM9c5w")
    gendata = SyntheticData(gemini=gemini)
    result = await gendata.generate_question_batch(chunks, 1)
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test())