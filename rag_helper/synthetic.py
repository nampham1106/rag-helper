import os
import instructor
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio as asyncio
from asyncio import Semaphore
from tenacity import retry, stop_after_attempt, wait_random_exponential
import google.generativeai as genai
from .gemini_helper import GeminiHelper



class QuestionAnswerResponse(BaseModel):
    """
    Đây là mô hình đại diện cho câu hỏi và câu trả lời mẫu bắt nguồn từ đoạn văn bản nhất định. Nó hữu ích trong việc giúp đỡ
    """

    chain_of_thought: str = Field(
        ...,
        description="Quá trình suy luận dẫn đến câu trả lời và câu hỏi được tạo ra.",
    )

    list_question_answer: list[dict[str, str]] = Field(
        ...,
        description="Danh sách các cặp câu hỏi và câu trả lời giả định mà người dùng có thể yêu cầu tìm kiếm để truy xuất đoạn văn bản đó. Đảm bảo sử dụng thông tin duy nhất cho chính đoạn văn bản đó và cũng giải thích mọi loại thông tin/từ viết tắt sử dụng trong câu trả lời."
    )
    
    
    # question: str = Field(
    #     ...,description="Câu hỏi được sinh ra từ đoạn văn bản."
    # )

    # answer: str = Field(...,description="Kết quả tóm tắt ra từ đoạn văn bản")

class MetaData(BaseModel):
    """
    This is a model which represents some metadata that we want to generate from a given text.

    Make sure to expand on the text by extracting out any accronyms, context or pharses that users might search for later on \
    when trying to retrieve this specific chunk and model the metadata in a way that allows us to retrieve the most relevant chunks when using searching for the query  
    """

    keywords: list[str] = Field(
        ...,
        description="This is a field which represents keywords that a user might use to search for this text",
    )
    hypothesis_pharses: list[str] = Field(
        ...,
        description="This is a field which represents hypothesis pharses that a user might use to search for this text",
    )

class SyntheticData():
    def __init__(self, gemini: GeminiHelper):
        self.clients = gemini.get_client()

    async def generate_question_batch(self,
        text_chunk_batch, max_concurrent_calls: int,
    ):
        sem = Semaphore(max_concurrent_calls)

        @retry(
            wait=wait_random_exponential(multiplier=1, min=10, max=90),
            stop=stop_after_attempt(3),
        )
        async def generate_question(text: str):
            async with sem:
                question = self.clients.messages.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "Bạn là một công cụ tìm kiếm đẳng cấp thế giới. Bạn sắp được chuyển một đoạn văn bản và công việc của bạn là tạo ra một cặp câu hỏi và câu trả lời giả định mà người dùng có thể yêu cầu tìm kiếm để truy xuất đoạn văn bản đó. Đảm bảo sử dụng thông tin duy nhất cho chính đoạn văn bản đó và cũng giải thích mọi loại thông tin/từ viết tắt sử dụng trong câu trả lời.",
                        },
                        {"role": "user", "content": f"Đây là đoạn văn bản: {text}"},
                    ],
                    response_model=QuestionAnswerResponse,
                    max_retries=3,
                )
                return (question, text)
        coros = [generate_question(item) for item in text_chunk_batch]
        res = await asyncio.gather(*coros)
        return [{"response": item, "source": text} for item, text in res]

    async def generate_metadata_batch(self,
        text_chunk_batch, max_concurrent_calls: int, 
    ):
        sem = Semaphore(max_concurrent_calls)

        @retry( 
            wait=wait_random_exponential(multiplier=1, min=10, max=90),
            stop=stop_after_attempt(3),
        )
        async def enhance_query(text_chunk: str):
            async with sem:
                return (
                    await self.client.messages.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a world class query indexing system. You are about to be passed a text chunk and you'll need to generate some metadata that will allow you to retrieve this specific chunk when the user makes a relevent query",
                            },
                            {"role": "user", "content": f"Here is the text chunk: {text_chunk}"},
                        ]
                    ),
                    text_chunk,
                )
        coros = [enhance_query(item) for item in text_chunk_batch]
        res = await asyncio.gather(*coros)
        return [{"response": item, "source": text} for item, text in res]