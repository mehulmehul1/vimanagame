# Asset Pipeline Quick Reference Guide
## VIMANA Production Workflow

---

## FILE STRUCTURE

```
docs/
├── asset-pipeline.yaml          # MASTER TRACKER - Update this!
├── asset-template.md            # Template for new assets
├── music-room-assets.md         # Detailed Music Room breakdown
└── ASSET-PIPELINE-GUIDE.md      # This file

docs/concepts/                   # Visual concepts per chamber
├── prologue/
├── hub/
├── culture/                     ← Music Room
├── art/
├── nature/
├── tech/
└── finale/

docs/references/                 # Reference boards
└── {chamber}/
```

---

## QUICK START: FOR TUHINANSHU

### Day 1: Getting Started
1. **Read** `docs/music-room-assets.md` - Your main work document
2. **Review** `docs/asset-pipeline.yaml` - See all assets and status
3. **Create** `docs/concepts/culture/` folders for your work
4. **Bookmark** This guide for reference

### Your First Task: Music Room Redesign
1. Open `docs/music-room-assets.md`
2. Go to Asset CULTURE-001 (Chamber Splat)
3. Follow the Pipeline Checklist
4. Update `asset-pipeline.yaml` as you progress

---

## PIPELINE STAGES (At a Glance)

| Stage | What You Do | Output | Update Tracker |
|-------|-------------|--------|----------------|
| **Storyboard** | Sketch ideas | Concepts folder | `storyboard: in-progress` |
| **References** | Collect images | References folder | `references: in-progress` |
| **Description** | Write details | Asset doc | `description: done` |
| **Concepts** | AI gen / sketches | Concepts folder | `concepts: in-progress` |
| **Model/Capture** | Create asset | Asset file | `modeling: in-progress` |
| **Integration** | Put in scene | Scene file | `integration: in-progress` |
| **Polish** | Final tweaks | Done! | `polish: done` → `status: done` |

---

## DAILY WORKFLOW

```
MORNING:
1. Open asset-pipeline.yaml
2. Find your assigned tasks
3. Check what's in progress

WORK:
4. Open the asset detail file (e.g., music-room-assets.md)
5. Follow the checklist for your asset
6. Save files in correct folders

EVENING:
7. Update asset-pipeline.yaml with your progress
8. Mark stages as done
9. Note any blockers
```

---

## UPDATING THE TRACKER

### When to Update
- ✅ After completing any pipeline stage
- ✅ When starting a new asset
- ✅ If you encounter a blocker
- ✅ At the end of each work day

### How to Update

In `docs/asset-pipeline.yaml`:

```yaml
your-asset:
  id: CULTURE-001
  status: in-progress        # Change this!
  pipeline:
    storyboard: done         # Mark stages done
    references: in-progress  # As you go
    # ...
```

### Status Values
- `backlog` → Not started
- `in-progress` → Actively working
- `review` → Ready for feedback
- `done` → Complete!

---

## ASSET TYPES QUICK REFERENCE

| Type | Description | Example | Output Folder |
|------|-------------|---------|---------------|
| `splat-env` | Gaussian Splat scene | Chamber environment | `public/assets/splats/` |
| `glb-model` | 3D model | Harp, props | `public/assets/models/` |
| `shader` | GLSL shader code | Water, vortex | `src/shaders/` |
| `vfx` | Visual effects | Post-processing | Code integration |
| `audio` | Sound/music | Harp notes | `public/assets/audio/` |
| `ui` | Interface elements | Shell UI | `src/ui/` |

---

## FILE NAMING CONVENTIONS

### Asset Files
```
public/assets/
├── splats/{chamber}/{scene-name}.splat
├── models/{chamber}/{asset-name}.glb
├── textures/{chamber}/{asset-name}/{texture-type}.png
└── audio/{chamber}/{sound-name}.mp3
```

### Concept Files
```
docs/concepts/{chamber}/{asset-name}/
├── storyboard.md
├── mood-board.md
├── sketch-01.png
├── ai-gen-01.png
└── final-direction.md
```

---

## COMMON TASKS

### Starting a New Asset
1. Copy `docs/asset-template.md`
2. Name it: `asset-{ID}-{name}.md`
3. Fill in the basics (ID, name, type, assigned)
4. Add to `asset-pipeline.yaml`

### Requesting Feedback
1. Update asset status to `review`
2. Post in team chat: "CULTURE-001 ready for review"
3. Wait for Mehul's feedback
4. Address feedback or mark as `done`

### When You're Blocked
1. Update asset status to `blocked`
2. Note what you're waiting on in `notes`
3. Tag Mehul in chat
4. Move to another asset if possible

---

## QUALITY CHECKLIST

### Before Marking "Done"

**For GLB Models:**
- [ ] Imports without errors
- [ ] Scale is correct (1 unit = 1 meter)
- [ ] Y-up orientation
- [ ] Poly count within target
- [ ] Textures are power-of-2
- [ ] Tested in Three.js viewer

**For Splat Scenes:**
- [ ] All quality variants captured
- [ ] File size under 50MB
- [ ] Tested in web viewer
- [ ] No major artifacts

**For Concepts:**
- [ ] Multiple variations explored
- [ ] Direction is clear
- [ ] Reference images collected
- [ ] Ready for production

---

## GIT WORKFLOW FOR TUHINANSHU

### Basic Commands
```bash
# See current status
git status

# See what changed
git diff

# Add your changes
git add docs/asset-pipeline.yaml
git add public/assets/models/culture/

# Commit
git commit -m "asset: add harp mechanism model WIP"

# Push
git push
```

### Branch Workflow
```bash
# Create feature branch
git checkout -b feature/music-room-assets

# Work on assets...
git add .
git commit -m "asset: chamber splat capture"

# Push branch
git push -u origin feature/music-room-assets

# Create PR on GitHub for review
```

---

## GETTING HELP

### For Design Questions
- Post in team chat with asset ID
- Attach concept images
- Ask specific questions

### For Technical Issues
- Check if asset type needs Mehul's review (shaders, code)
- Ask for help with file formats
- Report tool problems

### For Workflow Questions
- Check this guide first
- Ask Mehul for clarification

---

## ASSET SUMMARY

### Total Assets: 38

| Chamber | Environment | Props | Shaders | Audio | VFX |
|---------|-------------|-------|---------|-------|-----|
| Prologue | 1 | - | - | - | 1 |
| Hub | 1 | 1 | - | - | - |
| Culture (Music) | 1 | 3 | 3 | 3 | 2 |
| Art | 1 | 1 | - | - | - |
| Nature | 1 | 1 | - | - | - |
| Tech | 1 | 1 | - | - | - |
| Finale | - | - | 1 | - | 2 |

---

## TRACKING YOUR PROGRESS

### Weekly Checkpoints

**Week 1:**
- [ ] Music Room concepts approved
- [ ] References collected
- [ ] Chamber splat started

**Week 2:**
- [ ] Chamber splat complete
- [ ] Harp model started
- [ ] Jelly reference done

**Week 3:**
- [ ] All props modeled
- [ ] Shaders reviewed
- [ ] Integration started

**Week 4:**
- [ ] All assets integrated
- [ ] Testing complete
- [ ] Ready for next room

---

## HANDOFF CHECKLIST

### When an Asset is Complete
- [ ] All pipeline stages marked `done`
- [ ] Files in correct locations
- [ ] `asset-pipeline.yaml` updated
- [ ] Team notified in chat
- [ ] PR created (if code changed)

---

## QUOTE OF THE DAY

> "The ship doesn't test you. It teaches you."

Same goes for this pipeline—ask questions, learn, iterate. We're building something beautiful together.

---

**Last Updated:** 2026-01-26
**For Questions:** Ask Mehul or Scrum Master agent
