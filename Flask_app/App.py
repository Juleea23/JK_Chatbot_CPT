from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient, login
from transformers import T5Tokenizer, MT5ForConditionalGeneration, pipeline
from PIL import Image
import traceback
import torch
import os
import pdfplumber
import glob
import pytesseract
import spacy

# API-Login
HUGGINGFACEHUB_API_TOKEN = ""
login(HUGGINGFACEHUB_API_TOKEN)
app = Flask(__name__)

# Pfade
VECTOR_DB_PATH = r"C:\langchain-env\vector_db"
DATA_FOLDER = r"C:\Creative Prompting Techniques\rasa\pdfs"

# Modell für Embeddings und semantische Suche
#MODEL_NAME = "sentence-transformers/msmarco-MiniLM-L6-cos-v5" #Test, wurde dann doch nicht verwendet
MODEL_NAME = "BAAI/bge-m3"

embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)
similarity_model = SentenceTransformer(MODEL_NAME)

# 🔢 Ähnlichkeitsscore für semantische Suche (je niedriger, desto mehr Treffer)
SIMILARITY_THRESHOLD = 0.4  

# 📖 NLP-Modell für Lemmatisierung (automatische Wortstamm-Erkennung)
nlp = spacy.load("de_core_news_sm")

# 💡 Modell für Antwortgenerierung
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=HUGGINGFACEHUB_API_TOKEN)

def extract_text_from_pdf(pdf_path):
    """Extrahiert sauberen Text aus PDFs und entfernt Kopf-/Fußzeilen."""
    extracted_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = text.split("\n")
                cleaned_lines = [
                    line.strip() for line in lines
                    if not is_header_or_footer(line)  # Kopf-/Fußzeilen herausfiltern
                ]
                extracted_text.append("\n".join(cleaned_lines))
    
    return "\n".join(extracted_text)

def is_header_or_footer(text):
    """Filtert typische Kopf-/Fußzeilen (z. B. Seitenzahlen, Kapitelüberschriften)."""
    header_patterns = ["Seite ", "Kapitel ", "Inhaltsverzeichnis", "Quelle:", "©"]
    return any(pattern.lower() in text.lower() for pattern in header_patterns)

def extract_text_with_ocr(pdf_path):
    """Falls eine PDF-Seite nur ein Bild ist, wendet OCR an."""
    text = extract_text_from_pdf(pdf_path)  # Erst normale Extraktion versuchen
    if not text.strip():  # Falls nichts erkannt wurde, OCR verwenden
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                img = page.to_image()
                text += pytesseract.image_to_string(img)
    return text

def process_pdfs():
    """Liest alle PDFs ein, erstellt Embeddings und speichert sie in FAISS."""
    print("🔄 Keine bestehende Datenbank gefunden – starte Verarbeitung der PDFs...")
    
    embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    all_texts = []
    all_metadata = []

    pdf_files = glob.glob(os.path.join(DATA_FOLDER, "*.pdf"))
    print(f"📂 {len(pdf_files)} PDFs gefunden.")

    for pdf_path in pdf_files:
        print(f"📄 Verarbeite: {pdf_path}")
        party = os.path.basename(pdf_path).replace(".pdf", "").lower()

        text = extract_text_from_pdf(pdf_path)
        print(f"🔍 Extrahierter Text: {len(text)} Zeichen")

        if not text.strip():
            print(f"⚠️ Kein Text erkannt in {pdf_path} – versuche OCR...")
            text = extract_text_with_ocr(pdf_path)

        if text.strip():
            chunks = text_splitter.split_text(text)
            print(f"📌 {len(chunks)} Text-Chunks erstellt.")
            all_texts.extend(chunks)
            all_metadata.extend([{"source": party}] * len(chunks))

    if all_texts:
        print(f"🛠 Erstelle FAISS-Datenbank mit {len(all_texts)} Chunks...")
        db = FAISS.from_texts(all_texts, embedding, metadatas=all_metadata)
        db.save_local(VECTOR_DB_PATH)
        print("✅ PDF-Verarbeitung abgeschlossen und Datenbank gespeichert!")
    else:
        print("🚨 Keine Texte extrahiert! Prüfe deine PDFs.")

# Prüfe, ob der Index existiert – falls nicht, PDFs verarbeiten
if not os.path.exists(VECTOR_DB_PATH):
    process_pdfs()

db = FAISS.load_local(VECTOR_DB_PATH, embeddings=embedding, allow_dangerous_deserialization=True)

@app.route("/query", methods=["POST"])
def query():
    try:
        if db is None:
            return jsonify({"error": "Datenbank wurde nicht gefunden!"}), 500

        data = request.json
        question = data.get("question", "").strip().lower()
        topic = data.get("topic", "").strip().lower()
        party = data.get("party", "").strip().lower()

        # Automatische Lemmatisierung des Topics
        topic_lemma = " ".join([token.lemma_ for token in nlp(topic)])

        # Semantische Suche mit FAISS
        docs = db.similarity_search(question, k=10)

        # Debugging: Zeigt alle extrahierten Textstellen
        extracted_texts = [(doc.page_content, doc.metadata.get("source", "Unbekannt")) for doc in docs]
        print("\n🔎 DEBUG: Rohdaten aus FAISS (vor Filterung):")
        for i, (text, source) in enumerate(extracted_texts):
            print(f"{i+1}. [Quelle: {source}] {text[:300]}...")

        # Partei- und Themenfilterung mit semantischer Ähnlichkeit
        def text_passes_filters(text, source):
            source_lower = source.lower()

            # ✅ Partei-Filter
            if party and party not in source_lower:
                return False

            # ✅ Exakte Wortsuche als Backup
            if topic in text.lower() or topic_lemma in text.lower():
                print(f"✅ Direkte Übereinstimmung gefunden in: {text[:100]}...")
                return True

            # ✅ Semantische Ähnlichkeitsprüfung
            text_embedding = similarity_model.encode(text, convert_to_tensor=True)
            topic_embedding = similarity_model.encode(topic_lemma, convert_to_tensor=True)
            similarity_score = util.pytorch_cos_sim(topic_embedding, text_embedding).item()

            print(f"🔍 DEBUG: Ähnlichkeit zwischen '{topic}' (Lemma: {topic_lemma}) und Text = {similarity_score}")

            return similarity_score >= SIMILARITY_THRESHOLD  # Nur relevante Texte behalten

        # ➡️ Filterung anwenden
        filtered_texts = [text for text, source in extracted_texts if text_passes_filters(text, source)]

        # 🛠 Debugging: Zeigt gefilterte Textstellen
        print("\n🔍 DEBUG: Gefilterte Textstellen nach Partei & semantischer Ähnlichkeit:")
        for i, text in enumerate(filtered_texts):
            print(f"{i+1}. {text[:300]}...")

        # ❌ Falls kein relevanter Inhalt gefunden → Fehler ausgeben
        if not filtered_texts:
            return jsonify({"response": f"Ich konnte keine Wahlprogramminformationen zur Partei '{party}' und Thema '{topic}' finden."})

        # 📌 3️⃣ Nutzung von max 5 relevanten Treffen. KI generiert Antwort auf dessen Basis. 
        context = "\n\n".join([f"- {text}" for text in filtered_texts[:5]])  # Maximal 5 relevante Treffer nutzen

        # 🔍 Debug: Prompt an KI-Modell
        print("\n🛠 DEBUG: Prompt an KI-Modell:")

        prompt_text = f"""
        Du bist ein Assistent für politische Bildung. Fasse die Position der Partei '{party}' zum Thema '{topic}' in maximal 5 Sätzen zusammen, indem du folgende Regeln beachtest: 1. Nenne ausschließlich Informationen, die in den Quellen enthalten sind. 2. Beachte ausschließlich Sätze, die vollständig sind und deren Bedeutung du erkennen kannst. 3. Stelle sicher, dass unterschiedliche inhaltliche Aspekte des Themas getrennt dargestellt werden. 4. Verwende klare, sachliche Sprache und formuliere objektiv. 5. Nutze wichtige Schlüsselbegriffe aus den Quellen, um die Originalbedeutung beizubehalten. 6. Solltest du keine Infos zum Thema '{topic}' finden, schreib bitte „Tut mir leid, das Wahlprogramm enthält keine Infos zu ‚{topic}‘.“
        Auszüge aus dem Wahlprogramm:
        {context}
        Zusammenfassung:
        """

        print(prompt_text)

        # 🧠 KI-Aufruf
        response_obj = client.chat_completion(messages=[{"role": "user", "content": prompt_text}]) #geht nur für mistralai-modell als cloud client

        print("\n🛠 DEBUG: Antwort von Hugging Face API:")
        print(response_obj)
        
        # ✅ Überprüfung der API-Antwort
        if isinstance(response_obj, dict) and "choices" in response_obj:
            try:
               response_text = response_obj["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                response_text = "Fehler: Unerwartete API-Antwortstruktur."
        else:
            response_text = "Fehler: Keine gültige Antwort vom KI-Modell erhalten."
        
        print(response_text)

        # 📩 JSON-Antwort für Rasa
        response_json = {"response": response_text.strip()}
        print("\n📩 DEBUG: JSON-Antwort für Rasa:", response_json)
        return jsonify(response_json)

    except Exception as e:
        error_details = traceback.format_exc()
        print(error_details)
        return jsonify({"error": "Interner Serverfehler", "details": error_details}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
