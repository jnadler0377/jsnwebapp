from pydantic import BaseModel
from typing import List

class OutstandingLien(BaseModel):
    holder: str
    amount: str

class OutstandingLiensUpdate(BaseModel):
    outstanding_liens: List[OutstandingLien]
