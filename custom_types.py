import pydantic

class RAGChunk(pydantic.BaseModel):
    text: str
    page: int

class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[RAGChunk]
    source_id: str


class RAGUpsertResult(pydantic.BaseModel):
    ingested:int

class RAGRetrievedChunk(pydantic.BaseModel):
    text: str
    source: str
    page : int | None
    score: float

class RAGSearchResult(pydantic.BaseModel):
    contexts: list[RAGRetrievedChunk]
    sources: list[str]

class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts:int