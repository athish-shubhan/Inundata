from pydantic import BaseModel, Field

class PredictResponse(BaseModel):
    lat: float
    lon: float
    risk_score: float
    nearest_grid_cell: dict
    drivers: dict

class ExposedAssetsResponse(BaseModel):
    n_assets: int
    total_value_jpy: int
    assets: list[dict]

class AgentRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)

class AgentResponse(BaseModel):
    report_markdown: str
    tool_log: list[dict]
    mode: str
    latency_ms: float
    query: str

class IngestResponse(BaseModel):
    source: str
    n_rows: int
    ok: bool
    errors: list[str]
    warnings: list[str]
    meta: dict
