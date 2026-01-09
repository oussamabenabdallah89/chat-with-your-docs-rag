# Chat With Your Docs (Local RAG) — FastAPI + ChromaDB + Ollama + React (Vite)

## 1) Description
Ce projet est une application RAG (Retrieval-Augmented Generation) **100% locale** :

- Tu **importes** un PDF
- Le backend **découpe** le texte en *chunks* et l’indexe dans **ChromaDB**
- Quand tu poses une question, il récupère les passages les plus pertinents, puis appelle **Ollama** pour générer une réponse
- Le frontend affiche la réponse et les sources

---

## 2) Structure du projet

```text
chat-with-your-docs-rag/
  backend/
  frontend/
  run_local.bat
  stop_local.bat
  .gitignore
  README.md
```

---

## 3) Prérequis

- Windows 10/11
- Python 3.10+ (idéalement 3.11)
- Node.js 18+ (ou 20)
- Ollama installé et opérationnel

### Vérification rapide

```bash
python --version
node -v
npm -v
ollama --version
```

---

## 4) Installation (une seule fois)

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend

```powershell
cd ..\frontend
npm install
```

---

## 5) Configuration du frontend (.env)

Dans `frontend/`, crée un fichier `.env` (sans extension) :

```env
VITE_API_URL=http://127.0.0.1:8000
```

Dans `frontend/src/App.jsx`, assure-toi d’avoir :

```js
const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
```

---

## 6) Démarrer l’application

### Option A — Double-clic
Double-clique `run_local.bat`.

### Option B — Manuel

#### Backend

```powershell
cd backend
.\.venv\Scripts\activate
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

#### Frontend

```powershell
cd frontend
npm run dev
```

---

## 7) Vérifier que tout fonctionne

- Backend health : `http://127.0.0.1:8000/health`
- API docs (si activées) : `http://127.0.0.1:8000/docs`
- Frontend : l’URL affichée dans la console Vite (souvent `http://localhost:5173`)

---

## 8) Utilisation

1. Ouvre le frontend
2. Uploade un PDF
3. Attends l’indexation (création des *chunks*)
4. Pose tes questions
5. Lis la réponse et les sources affichées

---

## 9) Arrêter

- Ferme les fenêtres ouvertes par `run_local.bat`, **ou**
- Lance `stop_local.bat`

---

## 10) Dépannage rapide

- Le frontend ne prend pas en compte `.env` : stop + relance `npm run dev`
- CORS / mauvaise URL API : vérifie `VITE_API_URL` (`127.0.0.1` vs `localhost`)
- Port 5173 déjà utilisé : Vite choisira un autre port (ex. `5174`)
- Backend “Not Found” sur `/` : normal, utilise `/health`
- Erreur Ollama : démarre Ollama + vérifie qu’un modèle est disponible :

```bash
ollama list
```

---

## 11) Stack technique

- Backend : FastAPI
- Vector DB : ChromaDB
- Embeddings / LLM : Ollama (local)
- Frontend : React + Vite

## Demo

### 1) Import d’un PDF
![Upload PDF](docs/screenshots/upload.png)

### 2) Question / Réponse + Sources
![Q&A](docs/screenshots/qa.png)
