# 3DAgent — A4 Leaflet Content (Front & Back)

> Reference copy of all text used in `docs/leaflet.html`. Edit here first, then mirror into the HTML.
> Competition: MMU × Infineon (TM Group — *Life Made Easier*). Format: A4, front & back.
> Required sections (per brief): **Title · Novelty & Inventiveness · Usefulness & Application · Hardware Specification · Project Team Members · SDG**.

---

## TITLE

**3DAgent**
*AI-Powered Intelligent Navigation & Asset Management for 3D Spaces*

An embodied, conversational AI agent built on Matterport digital twins — turning passive 360° virtual tours into interactive spaces you can **talk to, scan, navigate, and maintain**.

---

## ABSTRACT (front, short)

3DAgent transforms static 3D digital twins into intelligent, semantically-aware environments by fusing **large language models (LLaMA 3.3 70B)**, **multimodal vision (Llama 4 Scout 17B + YOLOv8 + Grounding DINO)**, and the **Matterport SDK**. Users speak or type natural-language commands to navigate, auto-catalogue physical assets from a single screenshot, query inventory, and file maintenance reports pinned to an exact 3D location — replacing manual facility audits with a fast, accessible AI pipeline.

---

## PROBLEM & OBJECTIVES

**Problem.** Traditional 3D virtual spaces are *passive and disconnected* — users can look around but cannot ask questions, mark assets, or get guidance. Facilities managers spend hours manually cataloguing equipment from floor plans, and there is no natural way for non-technical staff to navigate, audit, or report faults inside a digital twin.

**Objectives.**
1. Map free-form natural-language and voice intent to navigation, query, and action inside a Matterport space.
2. Automate asset cataloguing from viewport screenshots using open-vocabulary computer vision.
3. Provide a closed-loop facility workflow: detect → tag → query → **report → triage → resolve**.
4. Keep the whole system accessible from a browser, with a graceful offline fallback.

---

## NOVELTY & INVENTIVENESS  *(required)*

1. **Digital Twin × LLM fusion.** Couples a Matterport 3D digital twin directly to a 70B-parameter LLM, so the camera *teleports* in response to reasoning — not menu clicks. A 14-class semantic router turns one chat box into navigation, vision, inventory, planning, and maintenance.
2. **Hybrid open-vocabulary vision.** Llama 4 Scout *names* everything in a frame (open vocabulary — "fume hood", "forklift", "fire extinguisher"), then local **YOLOv8 / YOLOv8-seg + Grounding DINO** ground each name with bounding boxes and **edge-fitting polygon outlines** — far more reliable than asking an LLM for coordinates. Items the LLM names but CV can't localise still appear, just without a highlight (clean fallback).
3. **ReAct agentic verification.** For planning prompts ("find a room for a 6-person meeting"), the agent *reasons* about the requirement, queries recorded inventory for candidate rooms, **teleports to each**, and uses live vision to **verify the room's current state** before recommending it — sense → think → act → verify.
4. **Voice-filed, location-pinned maintenance.** Say "report chair #1 has a broken leg" and the agent extracts the asset, infers severity, and pins the report to the exact sweep — later staff can fly straight to the faulty item and see it outlined.
5. **Resilient by design.** Cloud LLM/vision via Groq for speed; a **local BLIP VQA + YOLO** path keeps detection working with no cloud key.

---

## KEY FEATURES

- 🗣️ **Natural-Language & Voice Navigation** — "I want to cook" / "take me to the nearest fire extinguisher". Web Speech API enables hands-free control.
- 📷 **AI Vision Asset Detection** — screenshot a viewport; Scout + YOLOv8-seg + Grounding DINO detect, count, classify, **outline**, and auto-tag assets to inventory.
- 🧠 **14-Intent Semantic Router** — navigate, query, mark, activity, where-am-I, scan, auto-tag, floor plan, report issue, list problems, ReAct planning, and chat.
- 🛠️ **Maintenance Reporting & Triage** — worker/admin roles, severity ranking (low → critical), status lifecycle (open → assigned → in-progress → resolved), fly-to faulty equipment.
- 📦 **Smart Asset Inventory** — full CRUD, per-area counts, serial numbering, **scan history + change detection (diff)**, one-click **CSV / PDF** export for facilities teams.
- 🗺️ **Interactive Floor-Plan Minimap** — live position tracking, asset markers, export annotated plans as PNG/PDF.

---

## USEFULNESS & APPLICATION  *(required)*

**Who it helps:** facilities & operations managers, factory/warehouse supervisors, lab and hospital asset coordinators, real-estate and retail teams, and any non-technical staff who need to understand a building remotely.

**What it replaces:** manual clipboard audits, static PDF floor plans, spreadsheet asset registers, and disconnected maintenance tickets.

**Where it applies:**
- **Industrial & warehouse** — inventory forklifts, pallets, racks, and safety equipment; flag hazards instantly.
- **Corporate & education** — find a meeting room that actually fits 6 people (verified by live vision), audit office equipment.
- **Healthcare & labs** — catalogue specialised equipment by its real name and track condition over time.
- **Real estate & retail** — remote walkthroughs with conversational Q&A and instant inventory.

**Value:** hours of manual cataloguing collapse into a few screenshots; faults are reported and located in seconds; the whole space becomes queryable in plain language from any browser.

---

## SYSTEM ARCHITECTURE (SENSE → THINK → ACT)

- **SENSE — Multi-source input:** voice + chat commands, viewport screenshots, autonomous Matterport sweep images.
- **THINK — Flask multi-tenant backend & AI core:** semantic intent router (LLaMA 3.3 70B), hybrid vision (Scout 17B + YOLOv8-seg + Grounding DINO, BLIP fallback), SQLAlchemy relational store (users, spaces, assets, scans, reports, chat logs).
- **ACT — Embodied action:** SDK move-to-sweep navigation, asset outlining, ReAct verification, maintenance triage, context-aware chat response.

*(See the diagram in `docs/architecture_v2.svg` / the leaflet front page.)*

---

## TECH STACK

| Layer | Technology |
|---|---|
| AI – Language | **LLaMA 3.3 70B** (intent routing, chat) via Groq |
| AI – Vision (cloud) | **Llama 4 Scout 17B** (open-vocabulary naming, room labelling) via Groq |
| AI – Vision (local) | **YOLOv8 / YOLOv8-seg** (boxes + segmentation), **Grounding DINO** (open-vocab boxes), **BLIP VQA** (offline fallback) |
| ML runtime | **PyTorch**, Ultralytics, HuggingFace Transformers |
| Backend | **Flask + SQLAlchemy**, SQLite, FPDF2 (PDF export) |
| Frontend | HTML5 / CSS3, JavaScript ES6+, Jinja2, Web Speech API, Canvas API |
| 3D Platform | **Matterport SDK** (digital twin + sweep navigation) |

---

## HARDWARE SPECIFICATION  *(required)*

3DAgent is a browser-based platform; heavy language/vision inference runs in the cloud (Groq LPU), while optional local CV models can use a GPU.

**Capture device (input):**
- Matterport Pro2 / Pro3 3D camera (or compatible 360°/LiDAR capture device) to generate the digital twin.

**Server / development machine (recommended):**
- CPU: quad-core+ (Intel Core i7 / AMD Ryzen 7 or better)
- RAM: 16 GB (8 GB minimum)
- GPU: NVIDIA CUDA GPU, ≥ 6 GB VRAM for fast local YOLOv8-seg / Grounding DINO / BLIP inference *(CPU-only also works, slower)*
- Storage: ~5 GB for model weights (YOLOv8-seg ≈ 24 MB, YOLOv8 ≈ 22 MB, Grounding DINO base, BLIP)
- OS: Windows 11 / Linux; Python 3.10+

**Cloud inference:** Groq API serving LLaMA 3.3 70B + Llama 4 Scout 17B — no local GPU required for the LLM/VLM path.

**Client device:** any modern browser (Chrome/Edge) with WebGL and a microphone for voice input.

---

## RESULTS & DISCUSSION

- **Intent classification accuracy: 93.3%** over 30 varied user commands across the intent classes.
- **Asset detection MAE: 0.63** (avg) — Living Room 0.35, Kitchen 0.48, Bedroom 1.06.
- **Automated evaluation suite:** 7 evaluators (label resolution, vision parser, activity mapping, scan-diff, API latency — all 100% on unit cases; intent + ReAct parser as live-API integration tests). DB-only API endpoints respond in ~1–2 ms (p95).

---

## PROJECT TEAM MEMBERS  *(required)*

- **Project Leader / Developer:** Beckham Wong Yao Kang
- **Supervisor:** Dr. Goh Pey Yun
- **Institution:** Multimedia University (MMU)
- **In collaboration with:** Infineon Technologies · TM Group — *Life Made Easier*

---

## SUSTAINABLE DEVELOPMENT GOALS (SDG)  *(required)*

- **SDG 9 — Industry, Innovation & Infrastructure:** brings AI and digital-twin infrastructure to building and facility operations.
- **SDG 11 — Sustainable Cities & Communities:** smarter, safer, more accessible building management; faster hazard and fault reporting.
- **SDG 12 — Responsible Consumption & Production:** accurate asset registers and condition tracking extend equipment life and cut waste from lost/duplicated assets.
- **SDG 8 — Decent Work & Economic Growth:** location-pinned maintenance reporting improves worker safety and reduces downtime.

---

## CONCLUSION

3DAgent shows that frontier LLMs and computer vision can be fused with immersive 3D digital twins — using only existing models and open web standards — to create a platform where users **speak, scan, navigate, and maintain**. It replaces manual facility audits and static floor plans with an AI-driven pipeline that is fast, scalable, resilient, and accessible to non-technical users.
