from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    lat: float
    lon: float
    risk_score: float | None = Field(..., description="calibrated flood probability for the cell; null if the cell is not scored")
    risk_band: str
    status: str = Field(..., description="scored, open_water, cloud or no_data")
    grid_cell: dict
    drivers: dict
    driver_percentiles: dict = Field(..., description="share of scored cells in the AOI at or below this value, 0-100")
    observed_2020: dict


class ExposedAssetsResponse(BaseModel):
    n_assets: int
    total_value_jpy: int
    n_high_risk: int
    high_risk_value_jpy: int
    assets: list[dict]


class ExposedInfrastructureResponse(BaseModel):
    n_sites: int
    n_high_risk: int
    sites: list[dict]


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
