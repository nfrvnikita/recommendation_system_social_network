from sqlalchemy import Column, Integer, String, Text, desc, select

from database import Base, SessionLocal


class Post(Base):
    __tablename__ = "post"
    id = Column(Integer, primary_key=True)
    text = Column(Text)
    topic = Column(String, nullable=True)

    def __repr__(self):
        return f"{self.id} - {self.topic}"


if __name__ == "__main__":
    session = SessionLocal()
    print([
        x[0] for x in
        session.query(Post.id)
        .filter(Post.topic == "business")
        .order_by(desc(Post.id))
        .limit(10)
        .all()
       ]
    )
