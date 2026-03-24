"""
sql_rag_engine.py
-----------------
RAG pipeline: schema.sql -> LlamaIndex vector index -> HuggingFace chat -> SQL
"""

import os
import re
import traceback
from pathlib import Path
from typing import Optional

import sqlparse
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SQL_MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.3"
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
TOP_K_CHUNKS   = 6
MAX_NEW_TOKENS = 512
HF_TIMEOUT     = 60


# ---------------------------------------------------------------------------
# Schema parser  ->  one Document per CREATE TABLE + a summary doc
# ---------------------------------------------------------------------------
def _parse_schema_to_documents(schema_path: str):
    sql_text  = Path(schema_path).read_text(encoding="utf-8")
    documents = []

    for stmt_text in re.findall(r"(CREATE\s+TABLE\s+[\s\S]+?;)", sql_text, re.IGNORECASE):
        m = re.search(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", stmt_text, re.IGNORECASE)
        if not m:
            continue
        table_name = m.group(1)
        col_names  = re.findall(r"^\s{2,}(\w+)\s+", stmt_text, re.MULTILINE)
        columns    = ", ".join(col_names) if col_names else "unknown"

        documents.append(Document(
            text=(
                f"Table: {table_name}\n"
                f"Columns: {columns}\n\n"
                f"Full DDL:\n{stmt_text.strip()}"
            ),
            metadata={"table": table_name},
            excluded_embed_metadata_keys=["table"],
            excluded_llm_metadata_keys=["table"],
            id_=f"table_{table_name}",
        ))

    if not documents:
        documents = [Document(text=sql_text, metadata={"source": schema_path})]

    # Global summary doc
    summary = "DATABASE SCHEMA SUMMARY\n" + "\n".join(
        f"  • {d.metadata.get('table','?')}" for d in documents
    )
    documents.append(Document(
        text=summary,
        metadata={"table": "_summary"},
        excluded_embed_metadata_keys=["table"],
        excluded_llm_metadata_keys=["table"],
        id_="schema_summary",
    ))
    return documents


# ---------------------------------------------------------------------------
# Prompt builder  ->  returns (system_msg, user_msg) for chat_completion
# ---------------------------------------------------------------------------
def _build_chat_messages(user_question: str, schema_context: str):
    system_msg = (
        "You are an expert SQL developer. "
        "Given the database schema provided, write a single correct and efficient SQL query "
        "that answers the user request.\n"
        "Rules:\n"
        "- Return ONLY the raw SQL query.\n"
        "- No markdown fences (no ```sql).\n"
        "- No explanation before or after.\n"
        "- End the query with a semicolon."
    )
    user_msg = (
        f"## Database Schema\n{schema_context}\n\n"
        f"## Request\n{user_question}\n\n"
        "Write the SQL query now:"
    )
    return system_msg, user_msg


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
class SQLRAGEngine:
    def __init__(
        self,
        schema_path: str = "schema.sql",
        hf_token: Optional[str] = None,
        model_id: str = SQL_MODEL_ID,
        top_k: int = TOP_K_CHUNKS,
    ):
        self.schema_path = schema_path
        self.model_id    = model_id
        self.top_k       = top_k
        self.hf_token    = hf_token or os.getenv("HF_TOKEN", "")

        if not self.hf_token:
            raise ValueError("HuggingFace token is required. Set HF_TOKEN in .env")

        self._index     = None
        self._retriever = None

    # -- Build --
    def build_index(self):
        print(f"Loading schema: {self.schema_path}")
        docs = _parse_schema_to_documents(self.schema_path)
        print(f"Parsed {len(docs)} schema chunks")

        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_ID)
        Settings.embed_model   = embed_model
        Settings.llm           = None
        Settings.chunk_size    = 4096
        Settings.chunk_overlap = 128

        print("Building VectorStoreIndex…")
        self._index = VectorStoreIndex.from_documents(docs, show_progress=True)
        self._retriever = VectorIndexRetriever(index=self._index, similarity_top_k=self.top_k)
        print("Index ready!\n")

    # -- Retrieve --
    def retrieve_schema_context(self, question: str) -> str:
        if self._retriever is None:
            raise RuntimeError("Call build_index() first.")
        nodes = self._retriever.retrieve(question)
        return "\n\n---\n\n".join(n.get_content(metadata_mode="none") for n in nodes)

    # -- Call HF API --
    def _call_hf_api(self, system_msg: str, user_msg: str) -> str:
        client = InferenceClient(
            model=self.model_id,
            token=self.hf_token,
            timeout=HF_TIMEOUT,
            provider="hf-inference",   # force HF's own servers, skip novita/nscale/etc.
        )
        print(f"Calling {self.model_id}…")
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.01,
        )
        return response.choices[0].message.content.strip()

    # -- Clean SQL --
    @staticmethod
    def _clean_sql(raw: str) -> str:
        raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).replace("```", "")
        raw = re.sub(r"^\s*\[?SQL\]?\s*:?\s*", "", raw, flags=re.IGNORECASE)
        lines = []
        for line in raw.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.lower().startswith(("this query", "the query", "note:", "explanation")):
                break
            lines.append(line)
        sql = "\n".join(lines).strip()
        if sql and not sql.rstrip().endswith(";"):
            sql += ";"
        return sql

    # -- Generate --
    def generate(self, user_question: str) -> dict:
        if self._index is None:
            self.build_index()

        schema_context       = self.retrieve_schema_context(user_question)
        system_msg, user_msg = _build_chat_messages(user_question, schema_context)
        raw_output           = self._call_hf_api(system_msg, user_msg)

        print(f"Raw output:\n{raw_output}\n")
        sql = self._clean_sql(raw_output)

        return {
            "sql":         sql,
            "schema_used": schema_context,
            "prompt":      f"[SYSTEM]\n{system_msg}\n\n[USER]\n{user_msg}",
            "model":       self.model_id,
        }