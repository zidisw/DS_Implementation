import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv("E:/All About Programs/Laskar AI/DS_Implementation/submission2/data.csv", sep=';', encoding='windows-1252')

URL = "postgresql://postgres.mybfsltcugisqcirtxay:Ichaz781anakjokam*@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"

engine = create_engine(URL)
df.to_sql('studentperformance', engine)