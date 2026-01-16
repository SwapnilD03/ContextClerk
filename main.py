import logging
import httpx
import numpy as np
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import re
import os
import time
import datetime
import asyncio
from functools import lru_cache
from collections import defaultdict
from data_loader import load_and_chunk_pdf,embed_texts
from vector_db import QdrantStorage
from custom_types import RAGQueryResult,RAGSearchResult,RAGUpsertResult,RAGChunkAndSrc
from collections import OrderedDict

ANSWER_CACHE = OrderedDict()
MAX_CACHE_SIZE = 256

load_dotenv()

inngest_client = inngest.Inngest(
    app_id = "rag_app",
    logger = logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@lru_cache(maxsize=128)
def embed_question_cached(question: str):
    return embed_texts([question])[0]

@lru_cache(maxsize=512)
def embed_sentence_cached(sentence: str):
    return embed_texts([sentence])[0]

@lru_cache(maxsize=256)
def embed_context_cached(text: str):
    return embed_texts([text])[0]


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def aggregate_sources(chunks):
    source_pages = defaultdict(set)

    for c in chunks:
        if c.page is not None:
            source_pages[c.source].add(c.page)

    return {
        src: sorted(pages)
        for src, pages in source_pages.items()
    }



def compute_faithfulness(answer: str, contexts) -> float:
    """
    Simple lexical faithfulness score based on overlap
    between answer tokens and retrieved context tokens.
    """
    if not contexts:
        return 0.0

    context_text = " ".join(c.text for c in contexts).lower()
    answer_tokens = set(re.findall(r"\b\w+\b", answer.lower()))

    if not answer_tokens:
        return 0.0

    supported = sum(1 for t in answer_tokens if t in context_text)
    return supported / len(answer_tokens)




def chunk_overlap_score(answer: str, chunk_text: str) -> float:
    answer_tokens = tokenize(answer)
    chunk_tokens = tokenize(chunk_text)

    if not answer_tokens:
        return 0.0

    overlap = answer_tokens & chunk_tokens
    return len(overlap) / len(answer_tokens)


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 10]


def embed_sentences(sentences: list[str]) -> list[list[float]]:
    return embed_texts(sentences)



def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def semantic_coverage(answer: str,contexts,similarity_threshold: float = 0.30):
    """
    Returns:
      - coverage_ratio
      - covered_contexts
      - missing_contexts
    """
    if not contexts:
        return 0.0, [], contexts

    sentences = split_into_sentences(answer)
    if not sentences:
        return 0.0, [], contexts

    sentence_embeddings = [embed_sentence_cached(s) for s in sentences]


    covered = []
    missing = []
    context_emb = embed_context_cached(contexts[0].text)
    for c in contexts:
        
        is_covered = False
        for sent_emb in sentence_embeddings:
            sim = cosine_similarity(sent_emb, context_emb)
            if sim >= similarity_threshold:
                is_covered = True
                break

        if is_covered:
            covered.append(c)
        else:
            missing.append(c)

    coverage_ratio = len(covered) / len(contexts)
    return coverage_ratio, covered, missing


def build_retry_prompt(question, missing_contexts, previous_answer):
    MAX_RETRY_CHARS = 1200
    context_block = ""
    used = 0

    for c in missing_contexts:
        if used + len(c.text) > MAX_RETRY_CHARS:
            break
        context_block += c.text + "\n\n"
        used += len(c.text)

    return (
        "You previously answered the question, but some aspects were missing.\n\n"
        "Previous answer:\n"
        f"{previous_answer}\n\n"
        "Additional context:\n"
        f"{context_block}\n\n"
        "Task:\n"
        "- Address ONLY the missing aspects using the additional context.\n"
        "- Do NOT repeat information already stated.\n"
        "- Be concise and factual.\n\n"
        f"Question: {question}\n"
        "Supplemental answer:"
    )


def is_multi_part_question(question: str) -> bool:
    keywords = ["and", "both", "benefits", "challenges", "pros", "cons", "advantages", "disadvantages"]
    q = question.lower()
    return sum(1 for k in keywords if k in q) >= 2




async def embed_batch_async(batch: list[str]) -> list[list[float]]:
    """
    Run blocking embedding calls in a thread pool
    to avoid blocking the event loop.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, embed_texts, batch)



async def embed_in_parallel_batches(chunks,batch_size: int = 4,max_concurrency: int = 2) -> list[list[float]]:
    """
    Embed text chunks using micro-batching + controlled parallelism.
    Preserves order, limits concurrency, and retries safely.
    """

    semaphore = asyncio.Semaphore(max_concurrency)
    num_batches = (len(chunks) + batch_size - 1) // batch_size
    results: list[list[list[float]]] = [[] for _ in range(num_batches)]

    async def process_batch(batch_idx: int, batch: list[str]):
        assert all(isinstance(t, str) for t in batch), "Embedding input must be strings"

        async with semaphore:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    embeddings = await embed_batch_async(batch)
                    results[batch_idx] = embeddings
                    return
                except Exception as e:
                    logging.warning(
                        f"Embedding batch {batch_idx} failed "
                        f"(attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(1)

    tasks = []
    for i in range(0, len(chunks), batch_size):
        batch_idx = i // batch_size

        batch_chunks = chunks[i : i + batch_size]
        batch_texts = [c.text for c in batch_chunks]  # extract text only

        tasks.append(process_batch(batch_idx, batch_texts))


    await asyncio.gather(*tasks)
    if any(len(batch) == 0 for batch in results):
        raise RuntimeError("One or more embedding batches failed to produce vectors")


    # Flatten results in original order
    all_embeddings: list[list[float]] = []
    for batch_embeddings in results:
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)

async def rag_ingest_pdf(ctx:inngest.Context):

    def _load(ctx:inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id",pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks,source_id=source_id)

    async def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        logging.info(f"Number of chunks: {len(chunks)}")

        source_id = chunks_and_src.source_id
        
        try:
            # Parallel + micro-batched embeddings
            vectors = await embed_in_parallel_batches(
                chunks,
                batch_size=4,
                max_concurrency=2,
            )

            ids = [
                str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
                for i in range(len(chunks))
            ]

            payloads = [
            {
                "source": source_id,
                "text": chunks[i].text,
                "page": chunks[i].page,
            }
            for i in range(len(chunks))
            ]


            if not vectors:
                raise RuntimeError("No embeddings generated; aborting upsert")

            if len(vectors) != len(chunks):
                raise RuntimeError(
        f"Embedding count mismatch: {len(vectors)} vectors for {len(chunks)} chunks")


            QdrantStorage().upsert(ids, vectors, payloads)
            return RAGUpsertResult(ingested=len(chunks))

        except Exception as e:
            logging.error(f"Failed to embed/upsert: {str(e)}")
            raise


    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG:Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)

async def rag_query_pdf_ai(ctx:inngest.Context):

    def _search(question:str,top_k:int=5) -> RAGSearchResult:
        query_vec = embed_question_cached(question)
        

        store=QdrantStorage()
        found=store.search(query_vec,top_k)
        return RAGSearchResult(contexts=found["contexts"],sources=found["sources"])
    
    

    question = ctx.event.data["question"]

  
    top_k=int(ctx.event.data.get("top_k",5))

    cache_key = f"{question}|k={top_k}"


    if cache_key in ANSWER_CACHE:
        logging.info("cache_hit | returning cached answer")
        return ANSWER_CACHE[cache_key]


    t1 = time.time()
    
    found=await ctx.step.run("embed-and-search",lambda:_search(question,top_k),output_type=RAGSearchResult)
    t2 = time.time()

    # Guard against empty retrieval (anti-hallucination)
    if not found.contexts:
        return {
            "answer": "No relevant information found in the uploaded documents.",
            "sources": [],
            "num_contexts": 0,
            "confidence": 0.0,
            "grounded": False,
            "evidence": [],
        }
    
    # Sort by similarity score (highest first)
    contexts = sorted(found.contexts, key=lambda c: c.score, reverse=True)

    if not contexts or contexts[0].score < 0.25:
        return {
            "answer": "I couldn't find relevant information in the uploaded documents.",
            "confidence": 0.0,
            "grounded": False,
            "sources": {},
            "evidence": [],
            "num_contexts":0
        }
    

    MAX_CONTEXT_CHARS = 1800  
    top_score = contexts[0].score

    if top_score >= 0.85:
        MAX_CONTEXT_CHARS = 1500


    context_block = ""
    used_chars = 0
    selected_contexts = []

    multi_part = is_multi_part_question(question)

    for c in contexts:
        # Skip oversized chunks instead of stopping early
        if used_chars + len(c.text) > MAX_CONTEXT_CHARS:
            continue

        context_block += c.text + "\n\n"
        used_chars += len(c.text)
        selected_contexts.append(c)

        # For simple questions -> stop early
        if not multi_part and len(selected_contexts) >= 2:
            break

        



    user_content = (
    "You are answering a question using only the provided context.\n\n"
    "Instructions:\n"
    "- Answer clearly and accurately using the context below.\n"
    "- If the context partially answers the question, explain what is supported.\n"
    "- If the context does not contain the answer, say so explicitly.\n"
    "- Do not use external knowledge.\n\n"
    f"Context:\n{context_block}\n\n"
    f"Question: {question}\n\n"
    "Answer concisely and completely:"
   )
    
    OLLAMA_CLIENT = httpx.AsyncClient(
    base_url="http://localhost:11434",
    timeout=120
    )



    async def _llm_call(user_content: str):
        
            resp = await OLLAMA_CLIENT.post(
                "/api/chat",
                headers={
                    "Authorization": "Bearer ollama",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama3.2:latest",
                    "messages": [
                        {"role": "system", "content": "You answer questions using only the provided context."},
                        {"role": "user", "content": user_content}
                    ],
                    
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 20,
                    "repeat_penalty": 1.15,
                    "max_tokens": 180,
                    "stream":False
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]

    
    

    res = await ctx.step.run("llm-answer",  lambda: _llm_call(user_content))
    
    t3 = time.time()

    logging.info(
        f"timing | retrieval={t2-t1:.2f}s "
        f"llm={t3-t2:.2f}s"
    )




    answer = res.strip()

    COVERAGE_CHECK_K = min(3, len(contexts))

    coverage, covered_ctxs, missing_ctxs = semantic_coverage(
        answer,
        contexts[:COVERAGE_CHECK_K],
    )

    missing_ctxs = sorted(
    missing_ctxs,
    key=lambda c: c.score,
    reverse=True
    )[:2]



    is_partial = coverage < 0.65

    supplemental = None

    if is_partial and missing_ctxs:
        retry_prompt = build_retry_prompt(
            question,
            missing_ctxs,
            answer,
        )

        retry_res = await ctx.step.run(
            "llm-retry-missing",
            lambda: _llm_call(retry_prompt),
        )

        supplemental = retry_res["choices"][0]["message"]["content"].strip()

    if supplemental:
        answer = answer.rstrip() + "\n\n" + supplemental


    # Always compute metrics only on contexts the LLM actually saw
    used_contexts = list({id(c): c for c in (selected_contexts + missing_ctxs)}.values())


    # Safety guard (should rarely trigger)
    if not used_contexts:
        return {
            "answer": answer,
            "confidence": 0.0,
            "num_contexts":0,
            "grounded": False,
            "sources": {},
            "evidence": [],
        }
       
    # Evidence selection (deterministic)
    
    evidence_chunks = []

    for c in used_contexts:

        if c.page is None:
            continue

        overlap = chunk_overlap_score(answer, c.text)

        if overlap > 0.1:  # threshold to reduce noise
            evidence_chunks.append({
                "source": c.source,
                "page": c.page,
                "score": c.score,
                "overlap": overlap,
                
            })

    # Deduplicate evidence by (source, page), keep strongest overlap
    deduped = {}

    for e in evidence_chunks:
        key = (e["source"], e["page"])

        if key not in deduped or e["overlap"] > deduped[key]["overlap"]:
            deduped[key] = e

    # Final evidence list (ordered by overlap strength and keeping top-k)
    final_evidence = sorted(
        deduped.values(),
        key=lambda x: x["overlap"],
        reverse=True
    )
        

   
    

    source_pages = aggregate_sources(used_contexts)

    faithfulness = compute_faithfulness(answer, used_contexts)

    avg_retrieval_score = sum(c.score for c in used_contexts) / len(used_contexts)

    confidence = round(0.6 * faithfulness + 0.4 * avg_retrieval_score, 3)

    grounded = confidence >= 0.5


    res= {
    "answer": answer,
    "sources": source_pages,
    "num_contexts": len(used_contexts),
    "confidence": confidence,
    "grounded": grounded,
    "evidence": [
        {
            "source": e["source"],
            "page": e["page"],
        }
        for e in final_evidence
     ],
    }


    if len(ANSWER_CACHE) >= MAX_CACHE_SIZE:
        ANSWER_CACHE.popitem(last=False)

    if res["confidence"] > 0.6:    
        ANSWER_CACHE[cache_key] = res

    return res



app = FastAPI()




inngest.fast_api.serve(app,inngest_client,[rag_ingest_pdf,rag_query_pdf_ai]) 