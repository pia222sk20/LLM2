from typing import List, Literal
from pydantic import BaseModel,Field
class GradeDocuments(BaseModel):
    """문서 관련성 평가 스키마"""
    is_relevant: Literal["yes", "no"] = Field(
        description="문서가 질문과 관련이 있으면 'yes', 없으면 'no'"
    )

GradeDocuments(is_relevant = 'yes')
GradeDocuments(is_relevant = 'no')
GradeDocuments(is_relevant = 'maybe')