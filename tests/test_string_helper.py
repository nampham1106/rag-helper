from rag_helper.string_helpers import strip_punctuation

def test_strip_punctuation():
    s = "Hello, World!"
    print(strip_punctuation(s))

if __name__ == "__main__":
    test_strip_punctuation()