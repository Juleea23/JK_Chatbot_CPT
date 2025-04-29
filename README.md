AI-supported Chatbot for political questions – Project Report

This repository contains some of the related documents to the project report with the title "Political education through AI: development and reflection of an intelligent chatbot".
The project is about the development of an assistance system to low-threshold processing of election programs via RAG approach. The focus is on the combination of semantic search (via FAISS) and answer generation (via Hugging Face Inference API).

This Repository contains:
- **Flask-App** (`flask_app/app.py`)
- **Rasa** (`rasa_app/`)
- election programs have to be stored in form of PDFs in the folder rasa_app/pdfs
  
⚠️ For reasons of size and clarity not all documents have been uploaded in this repository. 

## ▶️ local execution
The document Requirements_Flask.txt contains a list of all necessary packages for the local execution of the Flask-App.
The document Requirements_Rasa.txt contains a list of all necessary packages for the local execution of the Rasa Chatbot.
These packages have to be installed in two different environments.  
The required LLMs can either be downloaded (if the device is powerful enough) or used with an cost-free API Token (as currently implemented in the code) via Huggingface.co.

## ▶️ further information
Further information to the technical implementation, limitations and ethical concerns as well as some output examples can be found in the related project report. 
For Questions or Feedback please contact: Jule.Koerbler@iu-study.org
