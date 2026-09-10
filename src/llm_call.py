import os
from dotenv import load_dotenv

import openai
from pydantic import BaseModel


load_dotenv()

client = openai.OpenAI(
     base_url="https://api.groq.com/openai/v1",
     api_key=os.environ["GROQ_API_KEY"],
)

# Call classic
reply = client.chat.completions.create(
    messages=[{"role": "user", "content": "Who is the best NLP teacher?"}],
    model="openai/gpt-oss-20b",
)
print(reply.choices[0].message.content)


# Call with structured output
class ComicName(BaseModel):
    name: str

reply = client.chat.completions.parse(
    messages=[{"role": "user", "content": "Extract the comedian's name in this video title: 'Ne me parlez plus d IA - la chronique de Thomas VDB'"}],
    model="openai/gpt-oss-20b",
    response_format=ComicName,
)

print(reply.choices[0].message.content)
