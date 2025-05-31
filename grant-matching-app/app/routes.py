from flask import Blueprint, request, render_template
from app.services.embedding_service import generate_embedding
from app.services.search_service import search_similar_grants
from app.services.generation_service import generate_summary

bp = Blueprint('main', __name__)

@bp.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["proposal"]
        embedding = generate_embedding(text)
        matches = search_similar_grants(embedding)
        results = [doc['content'] for doc in matches]
        response = generate_summary(text, results[0]) if results else "No match found."
        return render_template("index.html", response=response, results=results)
    return render_template("index.html")
