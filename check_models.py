import os

import google.generativeai as genai
from dotenv import load_dotenv


# Mirror main.py environment loading/config pattern.
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def main():
    for m in genai.list_models():
        name = getattr(m, "name", "")
        if "gemma" in name.lower():
            print(name)


if __name__ == "__main__":
    main()
