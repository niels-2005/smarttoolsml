import uuid
from datetime import datetime

import sqlalchemy.dialects.postgresql as pg
from sqlmodel import Column, Field, SQLModel, Relationship
from typing import List 
from src.books.schemas import Book

class User(SQLModel, table=True):
    __tablename__ = "user_accounts"

    uid: uuid.UUID = Field(
        sa_column=Column(
            pg.UUID,
            primary_key=True,
            unique=True,
            nullable=False,
            default=uuid.uuid4,
        )
    )

    role: str = Field(
        sa_column=Column(pg.VARCHAR, nullable=False, server_default="user")
    )

    username: str
    first_name: str
    last_name: str
    is_verified: bool = Field(default=False)
    email: str
    password_hash: str
    created_at: datetime = Field(sa_column=Column(pg.TIMESTAMP, default=datetime.now))
    updated_at: datetime = Field(sa_column=Column(pg.TIMESTAMP, default=datetime.now))

    books: List[Book] = Relationship( 
        back_populates="user", sa_relationship_kwargs={"lazy": "selectin"}
    )

    def __repr__(self) -> str:
        return f"<User {self.username}>"
