import instructor
import google.generativeai as genai

class GeminiHelper():
    def __init__(self, GOOGLE_API_KEY: str):
        genai.configure(api_key=GOOGLE_API_KEY)

        self.client = instructor.from_gemini(
            client=genai.GenerativeModel(
                model_name='models/gemini-1.5-flash-latest',
            ),
            mode=instructor.Mode.GEMINI_JSON,
        )
    
    def get_client(self):
        return self.client