# 🛸 Vimana Game Monorepo

Welcome to the **Vimana Game** development repository. This is a monolithic repository containing the game engine, editor, AI agents, and production documentation.

> [!NOTE]
> This project is built upon the **Shadow Engine** (originally by James C. Kane), extending its state-driven Gaussian Splatting architecture for a new narrative experience.

---

## 📂 Project Structure

This repository is organized into four main pillars to support full-stack game development and AI-assisted production.

| Component | Description | Location |
| :--- | :--- | :--- |
| **🎮 Vimana Game** | The primary game project, assets, and runtime logic. | [`/vimana`](file:///vimana) |
| **🛠️ Shadow Editor** | A professional PCUI-based web editor for scene management. | [`/editor`](file:///editor) |
| **🤖 AI Agents** | Production agents and autonomous workflows. | [`/_bmad`](file:///bmad) |
| **🧠 Brain Trust** | Agent outputs, brainstorming, and design history. | [`/_bmad-output`](file:///bmad-output) |

### Secondary Directories
- **`src/`**: The original "Shadow of the Czar" source code (retained for reference).
- **`docs/` & `engine-docs/`**: Core engine documentation and technical specifications.
- **`public/`**: Shared static assets and environment splats.

---

## 🚀 Getting Started

To get the environment running locally for development:

### 1. Run the Game (`Vimana`)
```bash
cd vimana
npm install
npm run dev
```
Explore the game at `http://localhost:5173`.

### 2. Run the Editor
```bash
cd editor
npm install
npm run dev
```
Access the development suite at `http://localhost:3001`.

---

## 🤝 Team Collaboration & AI Workflow

We use **BMAD (Behavioral Model Agentic Development)** to scale our production.

- **Agents**: Our specialized agents (Dev, Architect, QA) live in `/_bmad`.
- **Syncing**: All team members should track `_bmad-output`. This folder contains the "shared memory" of the project, including technical specs and sprint status.
- **Git Strategy**: Ensure you cross-reference [ENGINE_EDITOR_MAPPING.md](file:///ENGINE_EDITOR_MAPPING.md) when making core engine changes to avoid breaking editor compatibility.

---

## 📜 Documentation Index

- [Vimana Design Brief](file:///VIMANA_HARP_ENHANCED_DESIGN.md)
- [Implementation Summary (Phase 3)](file:///PHASE_3_SUMMARY.md)
- [Asset Pipeline Guide](file:///vimana/docs/ASSET-PIPELINE-GUIDE.md)

---

## 🛡️ License

Portions of this project are based on the Shadow Engine. This repository is for internal team development. See [LICENSE](file:///LICENSE) for core engine licensing details.
