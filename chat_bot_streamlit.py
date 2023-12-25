# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 15:35:20 2023

@author: alex-
"""

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

st.title("QA Chatbot with DistilBERT")

# Загрузка модели DistilBERT
model_name = "distilbert-base-uncased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Функция для ответа на вопрос пользователя
def get_answer(question, context):
    result = nlp(question=question, context=context)
    answer = result["answer"]
    score = result["score"]
    return answer, score

# Веб-интерфейс с помощью Streamlit
context = st.text_area("Введите контекст:")
question = st.text_input("Задайте вопрос:")

if st.button("Ответить"):
    if not context or not question:
        st.warning("Пожалуйста, введите контекст и вопрос!")
    else:
        answer, score = get_answer(question, context)
        st.markdown(f"**Ответ:** {answer}")
        st.markdown(f"**Уверенность:** {score}")