import os
from dotenv import load_dotenv

import openai
from pydantic import BaseModel


load_dotenv()

def main():
    provider = os.environ["LLM_PROVIDER"]

    client = get_client(provider)
    model_name = get_model_name(provider)

    print(" --- Making classic call --- ")
    reply = client.chat.completions.create(
        messages=[{"role": "user", "content": "Who is the best NLP teacher?"}],
        model=model_name,
    )
    print(reply.choices[0].message.content)
    
    
    print("")
    print(" --- Making call with response format --- ")
    # Call with structured output
    class ComicName(BaseModel):
        name: str
    
    reply = client.chat.completions.parse(
        messages=[{"role": "user", "content": "Extract the comedian's name in this video title: 'Ne me parlez plus d IA - la chronique de Thomas VDB'"}],
        # model="openai/gpt-oss-20b",
        model=model_name,
        response_format=ComicName,
    )
    
    print(reply.choices[0].message.content)
def get_model_name(provider):
    return {
        "GROQ": "openai/gpt-oss-20b",
        "OPEN_ROUTER": "nex-agi/nex-n2.5-mini:free",
    }[provider]


def get_client(provider):
    url = {
        "GROQ": "https://api.groq.com/openai/v1",
        "OPEN_ROUTER": "https://openrouter.ai/api/v1",
    }[provider]

    return openai.OpenAI(
         base_url=url,
         api_key=os.environ[f"{provider}_API_KEY"],
    )

if __name__ == "__main__":
    main()
