# llm/model.py
import os
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
    

def get_llm():
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 .env에 설정되지 않았습니다.")
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=SecretStr(api_key)     
            )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2")
        )

# 전역 LLM 인스턴스
llm = get_llm()
