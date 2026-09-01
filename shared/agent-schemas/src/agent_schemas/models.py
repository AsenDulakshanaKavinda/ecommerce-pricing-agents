
from datetime import date
from pydantic import BaseModel

SCHEMA_VERSION = "0.1.0"


class ForecastOutput(BaseModel):
    sku: str
    forecast_date: date
    predicted_units: float
    confidence_low: float
    confidence_high: float


class ReorderRecommendation(BaseModel):
    sku: str
    current_stock: int
    reorder_quantity: int
    urgency: str  # "low" | "medium" | "high"
    supplier_id: str
    reasoning: str


class CompetitorPriceSnapshot(BaseModel):
    sku: str
    competitor_name: str
    observed_price: float
    observed_at: date


class PricingRecommendation(BaseModel):
    sku: str
    current_price: float
    recommended_price: float
    min_margin_price: float
    reasoning: str


class DailyBrief(BaseModel):
    brief_date: date
    reorder_items: list[ReorderRecommendation]
    pricing_changes: list[PricingRecommendation]
    summary: str
    requires_approval: bool = True