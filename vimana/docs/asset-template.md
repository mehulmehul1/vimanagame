# Asset Template
# VIMANA Asset Production Template
# Copy this file for each new asset: asset-{ASSET-ID}-{NAME}.md

---

## Asset Information

| Field | Value |
|-------|-------|
| **Asset ID** | CULTURE-XXX |
| **Asset Name** | [Name] |
| **Chamber** | prologue / hub / culture / art / nature / tech / finale / global |
| **Asset Type** | splat-env / glb-model / shader / vfx / audio / ui / design |
| **Priority** | critical / high / medium / low |
| **Assigned To** | mehul / tuhinanshu / TBD |
| **Status** | backlog / in-progress / review / done / blocked |
| **Estimated Days** | X |

---

## Description

### What is this asset?

[Brief 2-3 sentence description of what this asset is and its role in the experience]

### Visual Reference

- **Mood Board:** [Link to Miro/Pinterest/Ref file]
- **Concept Art:** [File path or link]
- **Similar Real Objects:** [Reference links]

### Technical Notes

- **File Format:** [.splat / .glb / .glsl / .mp3 / etc]
- **Poly Count (if 3D):** [Target]
- **Texture Size:** [Target]
- **Special Requirements:** [Any technical constraints]

---

## Pipeline Checklist

### Stage 1: Storyboarding
- [ ] Visual concept established
- [ ] Placement in scene defined
- [ ] Interaction requirements clear
- [ ] Dependencies identified

**Output:** `docs/concepts/{chamber}/{asset-name}/storyboard.md`

### Stage 2: References
- [ ] Real-world reference images collected
- [ ] Art style references gathered
- [ ] Technical references (similar assets)
- [ ] Color palette defined

**Output:** `docs/references/{chamber}/{asset-name}.md`

### Stage 3: Description
- [ ] Detailed written description
- [ ] Material specifications
- [ ] Animation requirements (if any)
- [ ] Audio requirements (if any)

**Output:** Description in this file + AI prompt ready

### Stage 4: Concepts
- [ ] Initial sketches/compositions
- [ ] AI-generated variations (if applicable)
- [ ] Color studies
- [ ] Style exploration

**Output:** `docs/concepts/{chamber}/{asset-name}/`

### Stage 5: Modeling/Capture (depends on type)

**For Splat Environments:**
- [ ] Physical set prepared (if needed)
- [ ] Splat capture session
- [ ] Processing through Luma/Polycam/etc
- [ ] Quality variants generated (max/desktop/laptop/mobile)

**Output:** `public/assets/splats/{chamber}/{scene-name}.splat`

**For GLB Models:**
- [ ] Base mesh created
- [ ] UV unwrapping
- [ ] Texturing
- [ ] Material setup
- [ ] Export to GLB with proper scale/orientation

**Output:** `public/assets/models/{chamber}/{model-name}.glb`

**For Shaders:**
- [ ] Vertex shader written
- [ ] Fragment shader written
- [ ] Material class created
- [ ] Tested in scene

**Output:** `src/shaders/{shader-name}.{vert,frag}.glsl`

**For Audio:**
- [ ] Source recorded or synthesized
- [ ] Editing/processing
- [ ] Export to web-ready format
- [ ] Loop points set (if needed)

**Output:** `public/assets/audio/{chamber}/{sound-name}.mp3`

### Stage 6: Integration
- [ ] Asset imported to project
- [ ] Positioned in scene
- [ ] Lighting adjusted
- [ ] Interaction hooked up (if interactive)
- [ ] Tested across performance tiers

### Stage 7: Polish
- [ ] Visual feedback reviewed
- [ ] Performance validated (60fps desktop, 30fps mobile)
- [ ] Cross-browser testing
- [ ] Final tweaks applied

---

## Files

| File | Path | Status |
|------|------|--------|
| Concepts | `docs/concepts/...` | pending |
| References | `docs/references/...` | pending |
| Model/Splat | `public/assets/...` | pending |
| Textures | `public/assets/textures/...` | pending |
| Shaders | `src/shaders/...` | pending |
| Code | `src/entities/...` | pending |

---

## Dependencies

**This asset depends on:**
- [ ] {ASSET-ID} - {Reason}

**Other assets waiting on this:**
- [ ] {ASSET-ID} - {Reason}

---

## Notes

### Design Decisions
- [Record any important design choices made during production]

### Problems & Solutions
- [Document any issues encountered and how they were resolved]

### Feedback History
- **[Date]** - [Who]: [Feedback]
- **[Date]** - [Who]: [Response]

---

## Completion Criteria

Asset is considered DONE when:
- [ ] All pipeline stages complete
- [ ] Integrated in scene and tested
- [ ] Performance targets met
- [ ] Code review passed (if code asset)
- [ ] Design approved by Mehul

---

**Last Updated:** [Date]
**Last Updated By:** [Name]
