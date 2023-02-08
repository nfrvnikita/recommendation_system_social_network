from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, desc, select
from sqlalchemy.orm import relationship

from database import Base, SessionLocal
from table_post import Post
from table_user import User


class Feed(Base):
    __tablename__ = "feed_action"
    user_id = Column(Integer, ForeignKey(User.id), primary_key=True)
    user = relationship(User)
    post_id = Column(Integer, ForeignKey(Post.id), primary_key=True)
    post = relationship(Post)
    action = Column(String, nullable=False, primary_key=True)
    time = Column(DateTime, nullable=False, primary_key=True)


if __name__ == "__main__":
    session = SessionLocal()
    stmt = select(Feed.action, User.id).join(User).join(Post)
    session.execute(stmt).one()
