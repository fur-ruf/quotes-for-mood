from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tags import TAGS
import uvicorn
import os
from transformers import MarianMTModel, MarianTokenizer
import torch
from functools import lru_cache

try:
    ru_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    ru_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    ru_en_model.eval()
    
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.eval()
    
    tag_names = list(TAGS.keys())
    tag_texts = list(TAGS.values())
    tag_vectors = model.encode(tag_texts, normalize_embeddings=True)
    
except Exception as e:
    raise RuntimeError(f"Error loading models: {str(e)}")

app = FastAPI(title="Mood & Quote Tag Detector")

@lru_cache(maxsize=512)
def translate_ru_to_en(text: str) -> str:
    batch = ru_en_tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        generated = ru_en_model.generate(**batch)
    return ru_en_tokenizer.decode(generated[0], skip_special_tokens=True)

class TextInput(BaseModel):
    text: str

@app.post("/detect")
def detect_mood(data: TextInput):
    try:
        text = data.text[:300]
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        text_vector = model.encode([text], normalize_embeddings=True)
        similarities = cosine_similarity(text_vector, tag_vectors)[0]
        best_idx = int(np.argmax(similarities))

        return {
            "tag": tag_names[best_idx],
            "confidence": float(similarities[best_idx]),
            "top_3": sorted(
                [
                    {"tag": tag_names[i], "score": float(similarities[i])}
                    for i in range(len(tag_names))
                ],
                key=lambda x: x["score"],
                reverse=True
            )[:3]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/detect/ru")
def detect_mood_ru(data: TextInput):
    try:
        text_ru = data.text[:300]
        if not text_ru.strip():
            raise HTTPException(status_code=400, detail="The text cannot be empty")
        
        text_en = translate_ru_to_en(text_ru)
        text_vector = model.encode([text_en], normalize_embeddings=True)
        similarities = cosine_similarity(text_vector, tag_vectors)[0]
        best_idx = int(np.argmax(similarities))

        return {
            "original_text": text_ru,
            "translated_text": text_en,
            "tag": tag_names[best_idx],
            "confidence": float(similarities[best_idx]),
            "top_3": sorted(
                [
                    {"tag": tag_names[i], "score": float(similarities[i])}
                    for i in range(len(tag_names))
                ],
                key=lambda x: x["score"],
                reverse=True
            )[:3]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok", "tags_count": len(tag_names)}

@app.get("/")
def root():
    return {
        "message": "Mood & Quote Tag Detector API",
        "endpoints": ["/detect", "/detect/ru", "/docs", "/health"],
        "available_tags": tag_names
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)