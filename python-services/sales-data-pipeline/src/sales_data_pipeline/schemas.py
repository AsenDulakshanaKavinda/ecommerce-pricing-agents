import datetime

import pandera.pandas as pa
from pandera.typing.pandas import Series
from pydantic import BaseModel


class RawSalesRecord(BaseModel):
    """ Raw-level contract for a raw sales """
    date: str
    store: str
    item: int
    sales: int 

class StagedSalesSchema(pa.DataFrameModel):
    """ Dataframe contract after cleaning, debug and type convert. """
    date: datetime.datetime # can use - pa.DateTime
    item: int
    sales: int

    class Config:
        strict = True
        coerce = True


class CuratedSalesSchema(pa.DataFrameModel):
    """ Dataframe contract for the training-ready table forecast-agent consumes """
    date: datetime.datetime # can use - pa.DateTime
    item: int
    sales: int
    sales_lag_1: float
    sales_lag_7: float

    class Config:
        strict = True
        coerce = True