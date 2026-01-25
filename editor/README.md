# Shadow Web Editor

Professional web-based 3D game editor for the Shadow Czar Engine using PCUI (PlayCanvas UI Framework).

## Status

**Phase 1, Task 1: COMPLETED** ✅

- Project initialized with React + Vite + TypeScript
- PCUI framework integrated
- Basic 4-panel layout created
- TypeScript strict mode configured
- Path aliases configured for clean imports

## Project Structure

```
editor/
├── core/                 # Core editor systems (EditorManager, SelectionManager, etc.)
├── panels/               # Editor panels
│   ├── Viewport/        # 3D rendering area with Three.js + Spark.js
│   ├── Hierarchy/       # Scene tree view
│   ├── Inspector/       # Property editor
│   ├── AssetBrowser/    # Asset management
│   ├── Timeline/        # Animation timeline
│   ├── NodeGraph/       # Visual scripting
│   └── Console/         # Logs and debug output
├── components/          # Shared React components
├── data/                # Data models and TypeScript interfaces
├── utils/               # Utility functions
├── styles/              # Global styles
├── main.tsx             # Application entry point
├── App.tsx              # Root component
├── vite.editor.config.ts # Vite configuration
├── tsconfig.json        # TypeScript configuration
└── package.json         # Dependencies
```

## Tech Stack

- **Frontend**: React 18.3 with TypeScript
- **Build Tool**: Vite 5.4
- **UI Framework**: PCUI 5.5 (PlayCanvas UI)
- **3D Runtime**: Three.js 0.180 + @sparkjsdev/spark 0.1.10
- **State Management**: Observer pattern (PCUI)
- **Development**: Hot Module Replacement (HMR)

## Installation

The editor dependencies are already installed. If you need to reinstall:

```bash
cd editor
npm install
```

## Development

Start the development server:

```bash
cd editor
npm run dev
```

The editor will open at `http://localhost:3001`

## Available Scripts

- `npm run dev` - Start development server with HMR
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run type-check` - Run TypeScript type checking

## Configuration

### Path Aliases

The following path aliases are configured for clean imports:

- `@/core/*` → `editor/core/*`
- `@/panels/*` → `editor/panels/*`
- `@/components/*` → `editor/components/*`
- `@/data/*` → `editor/data/*`
- `@/utils/*` → `editor/utils/*`
- `@/styles/*` → `editor/styles/*`

Example usage:

```typescript
import { EditorManager } from '@/core/EditorManager';
import { Viewport } from '@/panels/Viewport/Viewport';
import { Button } from '@/components/Button';
```

### TypeScript Configuration

- **Strict mode**: Enabled
- **Target**: ES2020
- **Module system**: ESNext
- **Path mapping**: Configured for clean imports
- **Type checking**: Extensive (noUnusedLocals, noImplicitReturns, etc.)

## Architecture

The editor follows a modular architecture with clear separation of concerns:

### Core Systems (`/core`)

- **EditorManager**: Central state and coordination
- **SelectionManager**: Multi-object selection
- **UndoRedoManager**: Command pattern for undo/redo
- **ClipboardManager**: Copy/paste functionality
- **DataManager**: sceneData.js read/write

### Editor Panels (`/panels`)

Each panel is a self-contained React component:

- **Viewport**: Three.js canvas with gizmos and camera controls
- **Hierarchy**: TreeView of scene objects
- **Inspector**: Property editor with PCUI inputs
- **AssetBrowser**: GridView of project assets
- **Timeline**: Animation keyframe editor
- **NodeGraph**: Visual scripting editor
- **Console**: Log viewer and debug output

### Data Models (`/data`)

TypeScript interfaces for:
- Scene format (sceneData.js)
- Editor state
- Observer schemas
- Command objects

## Integration with Runtime

The editor integrates with the existing Shadow Czar Engine runtime:

- **Runtime location**: `../src/` (do not modify)
- **Data format**: sceneData.js
- **Managers**: GameManager, SceneManager, AnimationManager, etc.
- **Rendering**: Three.js + Spark.js Gaussian Splatting

## PCUI Integration

The editor uses PCUI for 80% of UI components:

```typescript
import { Panel, NumericInput, VectorInput, Observer, BindingTwoWay } from '@playcanvas/pcui';
import '@playcanvas/pcui/styles';

// Create observer for data binding
const observer = new Observer({
    position: { x: 0, y: 0, z: 0 }
});

// Two-way binding with UI
const input = new NumericInput({
    binding: new BindingTwoWay(),
    link: { observer, path: 'position.x' }
});
```

## Development Workflow

1. **Feature Development**: Create components in respective directories
2. **Type Safety**: Use TypeScript with strict mode
3. **State Management**: Use PCUI Observer for reactive data
4. **Styling**: Use PCUI built-in styles + custom CSS
5. **Testing**: Use npm run type-check before committing

## Current Limitations

- [ ] EditorManager not yet implemented
- [ ] Viewport canvas not yet created
- [ ] PCUI components not yet integrated
- [ ] Three.js scene not yet initialized
- [ ] Runtime integration not yet connected

## Next Steps (Phase 1, Task 2)

1. Implement EditorManager core class
2. Initialize Three.js scene in Viewport
3. Integrate PCUI components in panels
4. Add object selection system
5. Connect to existing runtime

## Planning

See the planning PRD for details:
- `../godot-web-editor/PRPs/shadow-web-editor-planning-prd.md`

## License

MIT

---

**Last Updated**: 2026-01-16
**Status**: Phase 1, Task 1 Complete
**Next Task**: Implement EditorManager and Viewport initialization
