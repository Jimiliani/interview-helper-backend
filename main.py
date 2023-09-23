from fastapi import FastAPI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tasks import TASKS

app = FastAPI()


@app.get("/")
async def root(text: str):
    response = []
    for task in TASKS:
        vectorizer = CountVectorizer().fit_transform([task["description"], text])

        cosine_sim = cosine_similarity(vectorizer)

        similarity_score = cosine_sim[0][1]
        response.append({"name": task["name"], "description": task["description"], "rating": f"{similarity_score:.3f}"})

    return {"response": response}
