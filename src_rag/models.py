from enum import Enum
import joblib
import numpy as np
import re
import tiktoken
import openai
import yaml

from FlagEmbedding import FlagModel

CONF = yaml.safe_load(open("config.yml"))

CLIENT = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=CONF["groq_key"],
)

tokenizer = tiktoken.get_encoding("cl100k_base")

MEMORY = joblib.Memory('.cache', verbose=0)

class MetadataAdded(Enum):
    NO_METADATA = "NO_METADATA"
    TITLE_AND_SECTION = "TITLE_AND_SECTION"


class QuestionEncoding(Enum):
    REGULAR = "REGULAR"
    HYDE = "HYDE"


def get_model(config):
    return RAG(**config["model"])


class RAG:
    def __init__(
            self,
            chunk_size=256,
            overlap=24,
            metadata_added=MetadataAdded.NO_METADATA,
            question_encoding=QuestionEncoding.REGULAR,
    ):
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._embedder = None
        self._loaded_files = set()
        self._texts = []
        self._chunks = []
        self._corpus_embedding = None
        self._client = CLIENT
        self._metadata_added = metadata_added
        self._question_encoding = question_encoding

    def load_files(self, filenames):
        texts = []
        for filename in filenames:
            if filename in self._loaded_files:
                continue

            with open(filename) as f:
                texts.append(f.read())
                self._loaded_files.add(filename)

        
        self._texts += texts

        chunks_added = self._compute_chunks(texts, self._metadata_added)
        self._chunks += chunks_added

        new_embedding = self.embed_corpus(chunks_added)
        if self._corpus_embedding is not None:
            self._corpus_embedding = np.vstack([self._corpus_embedding, new_embedding])
        else:
            self._corpus_embedding = new_embedding

    def get_corpus_embedding(self):
        return self._corpus_embedding

    def get_chunks(self):
        return self._chunks

    def embed_questions(self, questions):
        embedder = self.get_embedder()
        return embedder.encode(questions)

    def _compute_chunks(self, texts, metadata_added):
        return sum(
            (chunk_markdown_with_overlap(
                txt, chunk_size=self._chunk_size, overlap=self._overlap, metadata_added=metadata_added,
            ) for txt in texts),
            [],
        )

    def embed_corpus(self, chunks):
        embedder = self.get_embedder()
        return embedder.encode(chunks)

    def get_embedder(self):
        if not self._embedder:
            self._embedder = FlagModel(
                'BAAI/bge-base-en-v1.5',
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=True,
            )

        return self._embedder

    def reply(self, query: str) -> str:
        prompt = self._build_prompt(query)
        res = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
        )
        return res.choices[0].message.content
        

    def _build_prompt(self, query: str) -> str:
        context_str = "\n".join(self._get_context(query))

        return f"""Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
If the answer is not in the context information, reply \"I cannot answer that question\".
Query: {query}
Answer:"""

    def _get_context(self, query: str) -> list[str]:
        query_embedding = self.get_question_embedding(query, self._question_encoding, self._metadata_added)
        sim_scores = query_embedding @ self._corpus_embedding.T
        indexes = list(np.argsort(sim_scores[0]))[-5:]
        return [self._chunks[i] for i in indexes]
    
    def get_question_embedding(self, question: str, question_encoding: QuestionEncoding, metadata_added: MetadataAdded) -> np.array:
        if question_encoding == QuestionEncoding.HYDE:
            question = _get_hypothetical_document(question, metadata_added)

        return self.embed_questions([question])


@MEMORY.cache
def _get_hypothetical_document(question: str, metadata_added: MetadataAdded) -> str:
    """Generate a hypothetical document that would reply to the user question
    HyDE: Hypothetical Document Embedding"""
    if metadata_added == MetadataAdded.NO_METADATA:
        reply_format = "[document text]"
    elif metadata_added == MetadataAdded.TITLE_AND_SECTION:
        reply_format = """Title: [page title]
Section: [listing section and subsections like Section > subsection > subsubsection]

[document text]"""

    prompt = f"""Generate a fake document with the following format:
```md
{reply_format}
```
that would reply to the user query:
{question}"""
    res = CLIENT.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",
    )
    hypothetical_document = res.choices[0].message.content

    if "```md" in hypothetical_document:
        start = hypothetical_document.find("```md")
        hypothetical_document = hypothetical_document[start + 5:]

        if "```" in hypothetical_document:
            end = hypothetical_document.find("```")
            hypothetical_document = hypothetical_document[:end]

    return hypothetical_document


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def parse_markdown_sections(md_text: str) -> list[dict[str, str]]:
    """
    Parses markdown into a list of {'headers': [...], 'content': ...}
    Preserves full header hierarchy (e.g. ["Section", "Sub", "SubSub", ...])
    """
    pattern = re.compile(r"^(#{1,6})\s*(.+)$")
    lines = md_text.splitlines()

    sections = []
    header_stack = []
    current_section = {"headers": [], "content": ""}

    for line in lines:
        match = pattern.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()

            # Save previous section
            if current_section["content"]:
                sections.append(current_section)

            # Adjust the header stack
            header_stack = header_stack[:level - 1]
            header_stack.append(title)

            current_section = {
                "headers": header_stack.copy(),
                "content": ""
            }
        else:
            current_section["content"] += line + "\n"

    if current_section["content"]:
        sections.append(current_section)

    return sections


def sliding_window_chunk(tokens: list[int], chunk_size: int, overlap: int) -> list[list[int]]:
    step = chunk_size - overlap
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), step) if tokens[i:i + chunk_size]]


def chunk_markdown_with_overlap(
        md_text: str,
        chunk_size: int = 128,
        overlap: int = 24,
        metadata_added: MetadataAdded = MetadataAdded.NO_METADATA
) -> list[dict]:
    parsed_sections = parse_markdown_sections(md_text)
    chunks = []

    for section in parsed_sections:
        tokens = tokenizer.encode(section["content"])
        token_chunks = sliding_window_chunk(tokens, chunk_size, overlap)

        for token_chunk in token_chunks:
            chunk_text = tokenizer.decode(token_chunk)
            chunk_text = _add_metadata(chunk_text, section["headers"], metadata_added)
            chunks.append(chunk_text)

    return chunks

def _add_metadata(chunk_text: str, headers: list[str], metadata_added: MetadataAdded) -> str:
    if metadata_added == MetadataAdded.NO_METADATA:
        return chunk_text
    elif metadata_added == MetadataAdded.TITLE_AND_SECTION:
        title = headers[0]
        if len(headers) > 1:
            section_path = " > " .join(headers[1:])
        else:
            section_path = ""
                            
        return f"""Title: {title}
Section: {section_path}

{chunk_text}"""
