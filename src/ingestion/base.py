from dataclasses import dataclass, field
import geopandas as gpd

@dataclass
class IngestReport:
    source: str
    n_rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"source": self.source, "n_rows": self.n_rows, "ok": self.ok,
                "errors": self.errors, "warnings": self.warnings, "meta": self.meta}

@dataclass
class IngestResult:
    gdf: gpd.GeoDataFrame | None
    report: IngestReport
