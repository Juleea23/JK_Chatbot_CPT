AI-supported Chatbot for political questions – Project Report

This repository contains some of the related documents to the project report with the title "Political education through AI: development and reflection of an intelligent chatbot".
The project is about the development of an assistance system to low-threshold processing of election programs via RAG approach. The focus is on the combination of semantic search (via FAISS) and answer generation (via Hugging Face Inference API).

This Repository contains:
- **Flask-App** (`flask_app/langchain_api.py`): Schnittstelle zwischen Nutzereingaben, semantischer Suche und Antwortgenerierung.
- **Rasa-Aktionen** (`rasa_project/actions.py`): Beispielhafte benutzerdefinierte Rasa-Komponenten.
- **Beispieldaten/Promptsnippets** (optional): Zur Demonstration der Einbindung von Themen.

⚠️ Aus Gründen der Übersicht und Projektfokussierung sind nicht alle Dateien enthalten (z. B. Rasa-Konfig, Frontend, vollständige Chunks etc.).

## 🧠 Verwendete Technologien
- **Rasa** zur Dialogführung
- **LangChain & FAISS** zur Verwaltung und semantischen Suche in den Textdaten
- **Hugging Face Inference API** zur generativen Antworterstellung mit dem Modell `mistralai/Mistral-7B-Instruct-v0.3`
- **Flask** zur Bereitstellung der Schnittstelle

## ▶️ local execution
The document Requirements_LangChain.txt contains a list of all necessary packages for the local execution of the Flask-App.
The document Requirements_Rasa.txt contains a list of all necessary packages for the local execution of the Rasa Chatbot.
The packages have to be installed in two different environments.  
The required LLMs can either be downloaded (if the device is powerful enough) or used (as currently implemented in the code) with an cost-free API Token via Huggingface.co.
