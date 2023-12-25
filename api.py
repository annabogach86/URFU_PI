from fastapi import FastAPI as fapi
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from pydantic import BaseModel

class Item(BaseModel):
    context: str
    question: str

app = fapi()
model_name = "distilbert-base-uncased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("question-answering", model=model, tokenizer=tokenizer)

@app.get("/")
async def root():
    return {'message': 'Hello, World!'}

@app.post('/predict/')
def predict(item: Item):
    result = classifier(question=item.question, context=item.context)
    answer = result["answer"]
    score = result["score"]
    return answer, score