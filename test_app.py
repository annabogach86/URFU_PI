# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:07:31 2023

@author: alex-
"""


from fastapi.testclient import TestClient
from fastapi import HTTPException
from api import app, Item

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}

def test_predict():
    item = Item(context="The Universe contains everything that exists – not only the Earth and everything on it,but also all the planets, stars,
and galaxies, and the space in between them. The sun, at the center of the solar system, is just one of about 100 billion stars
in our galaxy, or collection of stars, called the Milky Way. Astronomers have estimated that there are about 100 billion other
galaxies in the universe. Most scientists think that the universe formed about 15 billion years ago in an enormous explosion
called the big bang.They also think that the universe is expanding.", question="What is at the center of the solar system?")
    response = client.post("/predict/", json=item.dict())
    
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "score" in response.json()

def test_predict_invalid_input():
    # Test case with missing context
    invalid_item = {"question": "What is at the center of the solar system?"}
    response = client.post("/predict/", json=invalid_item)
    assert response.status_code == 422  # HTTP 422 Unprocessable Entity (validation error)

    # Test case with missing question
    invalid_item = {"context": "The Universe contains everything that exists"}
    response = client.post("/predict/", json=invalid_item)
    assert response.status_code == 422

    # Test case with empty payload
    empty_item = {}
    response = client.post("/predict/", json=empty_item)
    assert response.status_code == 422

def test_predict_exception():
    # Test case where the question-answering model raises an exception
    def mock_classifier(*args, **kwargs):
        raise Exception("Mocked exception")

    app.classifier = mock_classifier
    item = Item(context="The Universe contains everything that exists – not only the Earth and everything on it,but also all the planets, stars,
and galaxies, and the space in between them. The sun, at the center of the solar system, is just one of about 100 billion stars
in our galaxy, or collection of stars, called the Milky Way. Astronomers have estimated that there are about 100 billion other
galaxies in the universe. Most scientists think that the universe formed about 15 billion years ago in an enormous explosion
called the big bang.They also think that the universe is expanding.", question="What is at the center of the solar system?")
    response = client.post("/predict/", json=item.dict())
    assert response.status_code == 500  # HTTP 500 Internal Server Error
