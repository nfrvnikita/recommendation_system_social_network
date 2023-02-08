from sqlalchemy import Column, Integer, String, Text, desc, select
from database import Base, SessionLocal


class Post(Base):
    __tablename__ = "post"
    id = Column(Integer, primary_key=True)  
    text = Column(Text)
    topic = Column(String, nullable=True)

    def __repr__(self):
        return f"{self.id} - {self.topic}"
