from datetime import UTC, datetime
import uuid

from sqlalchemy import Float, Text, Integer, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    ...
    

class RawDataset(Base):

    __tablename__ = "raw_dataset"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    invoice	: Mapped[str] = mapped_column(Text)
    stockCode: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    quantity: Mapped[int] = mapped_column(Integer)		
    invoiceDate: Mapped[datetime] = mapped_column(Text)
    price: Mapped[float] = mapped_column(Float)
    customer_id: Mapped[float] = mapped_column(Float)
    country: Mapped[str] = mapped_column(Text)	
