# 🌙 Moondream3 Gradio UI

Eine Web-Oberfläche für das Moondream3 Vision-Language-Modell mit Unterstützung für Bildbeschreibungen, visuelle Fragen & Antworten, Objekterkennung und Objekt-Pointing.

A web interface for the Moondream3 vision-language model featuring image captioning, visual question answering, object detection, and object pointing.

## Features / Funktionen

- **📝 Image Captioning / Bildbeschreibung**: Generiere beschreibende Texte für deine Bilder (kurz, normal, lang)
- **❓ Visual Q&A / Visuelle Fragen**: Stelle Fragen zu deinen Bildern und erhalte intelligente Antworten
- **🔍 Object Detection / Objekterkennung**: Erkenne und lokalisiere spezifische Objekte mit Bounding Boxes
- **👆 Object Pointing / Objekt-Pointing**: Zeige auf spezifische Objekte in deinen Bildern

## Requirements / Voraussetzungen

- Python 3.10+
- NVIDIA GPU mit CUDA empfohlen (ca. 19 GB VRAM für volle Leistung)
- Funktioniert auch auf CPU/MPS, aber langsamer

## Installation

### 1. Repository klonen oder Dateien herunterladen

```bash
cd MoonDream3
```

### 2. Virtuelles Environment erstellen (empfohlen)

```bash
python -m venv .venv

# Windows (CMD)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

### 3. PyTorch installieren

PyTorch muss separat installiert werden, da die Installation von deiner Hardware abhängt. Besuche [pytorch.org/get-started](https://pytorch.org/get-started/locally/) oder nutze einen der folgenden Befehle:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

## Verwendung / Usage

### Starten der Anwendung

```bash
python app.py
```

Die Anwendung startet und zeigt eine URL an (standardmäßig `http://127.0.0.1:7860`).

### Schritte zur Nutzung

1. **Öffne die URL** im Browser
2. **Klicke "Load Model"** um Moondream3 zu laden (kann beim ersten Mal etwas dauern, da das Modell heruntergeladen wird)
3. **Wähle einen Tab** für die gewünschte Funktion:
   - **Image Captioning**: Bild hochladen, Länge wählen, "Generate Caption" klicken
   - **Visual Q&A**: Bild hochladen, Frage eingeben, "Ask Question" klicken
   - **Object Detection**: Bild hochladen, Objekttyp eingeben (z.B. "person", "car"), "Detect Objects" klicken
   - **Object Pointing**: Bild hochladen, Objekttyp eingeben, "Point to Objects" klicken

## Öffentliches Teilen

Um die Anwendung öffentlich zugänglich zu machen (z.B. für Demos), ändere die letzte Zeile in `app.py`:

```python
demo.launch(share=True)
```

## Alternative: Moondream Cloud API

Wenn du keine lokale GPU hast, kannst du auch die Moondream Cloud API nutzen. Ändere dazu den Modell-Ladecode in `app.py`:

```python
import moondream as md

# Statt AutoModelForCausalLM.from_pretrained(...)
model = md.vl(api_key="DEIN_API_KEY")
```

Hole dir deinen API-Key im [Moondream Dashboard](https://moondream.ai).

## Troubleshooting

### Out of Memory (OOM)
- Versuche das Modell auf CPU zu laden (langsamer aber weniger VRAM)
- Schließe andere GPU-intensive Anwendungen

### Modell lädt nicht
- Stelle sicher, dass `transformers>=4.44.0` installiert ist
- Prüfe deine Internetverbindung (das Modell wird von Hugging Face heruntergeladen)

### Langsame Inferenz
- GPU wird empfohlen für schnelle Ergebnisse
- Das erste Laden und Kompilieren dauert länger, danach ist es schneller

## Lizenz

Siehe die [Moondream3 Model Card](https://huggingface.co/moondream/moondream3-preview) für Lizenzinformationen.

---

*Powered by [Moondream3](https://huggingface.co/moondream/moondream3-preview) & [Gradio](https://gradio.app)*
