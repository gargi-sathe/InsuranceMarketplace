#multiple PDFs
from flask import Flask, request, jsonify, render_template, session
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from openai import AzureOpenAI
from flask_cors import CORS
from flask_session import Session

# === Setup === #
app = Flask(__name__)
app.secret_key = "super-secret-key"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
CORS(app)

# === Azure OpenAI Configuration === #
DEPLOYMENT_NAME = "VARELab-GPT4o"  # Replace with your Azure deployment name
client = AzureOpenAI(
    api_key="",  # Replace with your Azure API Key
    api_version="2024-08-01-preview",
    azure_endpoint="https://vare-labs-azure-openai-resource.openai.azure.com"
)

# === Multiple PDFs to Load === #
PDF_FILES = ["Medicare_EOC.pdf", "El_Paso_EOC.pdf", "Molina_EOC.pdf"]  # Add all your PDF files here

# === Load PDF and Tag Chunks by Filename === #
def load_pdf_text(path):
    doc = fitz.open(path)
    chunks = []
    for page in doc:
        text = page.get_text()
        chunks += [t.strip() for t in text.split('\n\n') if len(t.strip()) > 100]
    return chunks

def load_all_pdfs(pdf_files):
    all_chunks = []
    for path in pdf_files:
        chunks = load_pdf_text(path)
        tagged = [f"[{os.path.basename(path)}]\n{chunk}" for chunk in chunks]
        all_chunks.extend(tagged)
    return all_chunks

pdf_chunks = load_all_pdfs(PDF_FILES)

# === Embedding and Indexing === #
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(pdf_chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# === Semantic Search === #
def get_top_chunks(query, k=5):
    query_embedding = embedder.encode([query])
    _, indices = index.search(query_embedding, k)
    return [pdf_chunks[i] for i in indices[0]]

# === Flask Routes === #
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        if "chat_history" not in session:
            session["chat_history"] = []

        context = "\n\n".join(get_top_chunks(user_message))

        messages = [{"role": "system", "content": "You are a helpful assistant answering questions based on the PDFs provided."}]
        for msg in session["chat_history"]:
            messages.append({"role": "user", "content": msg["user"]})
            messages.append({"role": "assistant", "content": msg["bot"]})

        messages.append({"role": "user", "content": f"Answer this using the context below:\n\n{context}\n\nQuestion: {user_message}"})

        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages
        )

        bot_reply = response.choices[0].message.content.strip()
        session["chat_history"].append({"user": user_message, "bot": bot_reply})
        session.modified = True

        return jsonify({"reply": bot_reply})

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/clear", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True)
