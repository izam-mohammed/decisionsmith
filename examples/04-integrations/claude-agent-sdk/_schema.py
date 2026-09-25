"""The schema the examples in this folder decide."""

from pydantic import BaseModel, Field


class Risky(BaseModel):
    is_destructive: bool = Field(description="Could this shell command delete data or change the system?")
