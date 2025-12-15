from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from .database import Base
import json

class Case(Base):
    __tablename__ = "cases"
    id = Column(Integer, primary_key=True, index=True)
    case_number = Column(String, unique=True, index=True, nullable=False)
    filing_datetime = Column(String, default="")
    style = Column(String, default="")
    division = Column(String, default="")
    judge = Column(String, default="")
    parcel_id = Column(String, default="")
    address = Column(String, default="")
    address_override = Column(String, default="")
    subscriber_case_link = Column(String, default="")
    verified_complaint_path = Column(String, default="")
    value_calc_path = Column(String, default="")
    mortgage_path = Column(String, default="")
    current_deed_path = Column(String, default="")
    previous_deed_path = Column(String, default="")
    appraiser_doc1_path = Column(String, default="")
    appraiser_doc2_path = Column(String, default="")
    arv = Column(Float, default=0.0)
    rehab = Column(Float, default=0.0)
    closing_costs = Column(Float, default=0.0)
    outstanding_liens = Column(Text, default="[]", nullable=False)  # JSON: [{"holder":"", "amount":""}]

    defendants = relationship("Defendant", back_populates="case", cascade="all, delete-orphan")
    dockets = relationship("Docket", back_populates="case", cascade="all, delete-orphan")

    # Helpers for JSON field
    def get_outstanding_liens(self):
        try:
            return json.loads(self.outstanding_liens or "[]")
        except Exception:
            return []

    def set_outstanding_liens(self, liens_list):
        try:
            self.outstanding_liens = json.dumps(liens_list or [])
        except Exception:
            self.outstanding_liens = "[]"

class Defendant(Base):
    __tablename__ = "defendants"
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"))
    name = Column(String, default="")
    case = relationship("Case", back_populates="defendants")

class Docket(Base):
    __tablename__ = "dockets"
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"))
    docket_datetime = Column(String, default="")
    docket_text = Column(Text, default="")
    link = Column(String, default="")
    case = relationship("Case", back_populates="dockets")

class Note(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), index=True, nullable=False)
    content = Column(Text, default="")
    created_at = Column(String, default="")
