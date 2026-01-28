# STORY-XXX-000: [Story Title]

**Epic**: `EPIC-XXX` - [Epic Title]
**Story ID**: `STORY-XXX-000`
**Points**: `TBD` (1, 2, 3, 5, 8, or 13)
**Status**: `Draft` | `Ready for Dev` | `In Dev` | `Review` | `Done`
**Owner**: `TBD`
**Created**: `YYYY-MM-DD`

---

## User Story

As a **[role]**, I want **[feature]**, so that **[benefit]**.

---

## Overview

[Brief description of what this story implements. 2-3 sentences.]

**Reference:**
- [Source Link](URL) - Description
- [Source Link](URL) - Description

---

## Technical Specification

### Key Technical Concepts

```typescript
// Example code snippet
interface ExampleInterface {
    property: type;
    method(): returnType;
}
```

```
┌──────────────────────────────────────────────────────────────┐
│  Process Flow Diagram                                        │
│                                                              │
│  Input → Process → Output                                    │
└──────────────────────────────────────────────────────────────┘
```

### Configuration/Constants

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `paramName` | value | min-max | Description |

---

## Implementation Tasks

1. **[TAG]** Task description
2. **[TAG]** Task description
3. **[TAG]** Task description

**Tags:** SYSTEM, SHADER, COMPUTE, RENDER, PIPELINE, UI, AUDIO, INTEGRATION, etc.

---

## File Structure

```
src/path/to/feature/
├── MainFile.ts
├── SubSystem.ts
└── types.ts
```

---

## Class/System Interface

```typescript
export class SystemName {
    constructor(
        private device: GPUDevice,
        private dependency: DependencyType
    ) {}

    // Public methods
    update(deltaTime: number): void {
        // Implementation
    }

    render(encoder: GPUCommandEncoder): void {
        // Implementation
    }

    // Private methods
    private helperMethod(): void {
        // Implementation
    }
}
```

---

## Shader Code (if applicable)

```wgsl
// Shader code example
@group(0) @binding(0) var<uniform> uniformData: UniformDataType;

fn main() {
    // Implementation
}
```

---

## Integration Points

```typescript
// In existing system
export class ExistingSystem {
    private newSystem: SystemName;

    update(deltaTime: number): void {
        // NEW: Integrate new system
        this.newSystem.update(deltaTime);
    }
}
```

---

## Acceptance Criteria

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3
- [ ] Performance: <X ms per frame
- [ ] Debug view: `window.debugVimana.feature.method()`

---

## Dependencies

- **Requires**: [Other story/system]
- **Required By**: [Other story/system]
- **Blocks**: [Other story/system]

---

## Edge Cases

1. **Edge case 1**: Description and handling
2. **Edge case 2**: Description and handling

---

## Notes

- Any additional notes
- Performance considerations
- Future improvements

---

**Sources:**
- [Source 1](URL)
- [Source 2](URL)
