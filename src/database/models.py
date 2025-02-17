from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

Base = declarative_base()

class Experiment(Base):
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False)
    peft_method = Column(String, nullable=False)
    learning_rate = Column(Float, nullable=False)
    batch_size = Column(Integer, nullable=False)
    num_epochs = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    metrics = relationship("TrainingMetric", back_populates="experiment")

class TrainingMetric(Base):
    __tablename__ = 'training_metrics'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    epoch = Column(Integer, nullable=False)
    loss = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    experiment = relationship("Experiment", back_populates="metrics")

# Database setup function
def init_db():
    engine = create_engine(os.environ["DATABASE_URL"])
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()
