# Development Setup - First Principles Guide

## Overview

**Development Setup** is the process of configuring your local machine to work on the Shadow Engine. This includes installing dependencies, setting up the development environment, configuring tools, and ensuring everything works together. A proper development setup makes the difference between frustrating debugging sessions and productive creation.

Think of development setup like the **"kitchen preparation"** before cooking—like a chef organizes their tools and ingredients before starting, a developer needs their environment properly configured to work efficiently.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Remove technical barriers from the creative process. When developers can quickly test changes and iterate on gameplay, they spend more energy making the game fun and less time fighting with their tools.

**Why Good Development Setup Matters?**
- **Fast Iteration**: Test changes in seconds, not minutes
- **Reliable Builds**: Consistent behavior across machines
- **Easy Collaboration**: New developers can start contributing quickly
- **Good Habits**: Linting and formatting catch bugs early
- **Professional Quality**: Team maintains consistent code style

---

## 🛠️ Prerequisites

### Required Software

Before starting, ensure you have:

| Tool | Version | Purpose |
|------|---------|---------|
| **Node.js** | 18+ | JavaScript runtime |
| **npm** | 9+ | Package manager |
| **Git** | Latest | Version control |
| **Code Editor** | VS Code recommended | Writing code |

### Check Your Versions

```bash
# Check Node.js (should be 18+)
node --version

# Check npm (should be 9+)
npm --version

# Check Git
git --version
```

---

## Step-by-Step Setup

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/shadow-engine.git
cd shadow-engine

# Or your preferred method
```

### 2. Install Dependencies

```bash
# Install all dependencies
npm install

# This creates:
# - node_modules/ (all packages)
# - package-lock.json (locked versions)
```

### 3. Verify Installation

```bash
# Run the development server
npm run dev

# You should see:
# > VITE v5.x.x ready in [time]
# > Local: http://localhost:3000/
```

### 4. Open in Browser

Navigate to `http://localhost:3000/` and verify:
- The game loads without errors
- Console shows no critical errors
- Graphics render correctly

---

## Recommended VS Code Setup

### Install Extensions

```bash
# From VS Code, install these extensions:
code --install-extension dbaeumer.vscode-eslint
code --install-extension esbenp.prettier-vscode
code --install-extension usernamehw.errorlens
code --install-extension ms-vscode.vscode-typescript-next
```

### VS Code Settings

Create `.vscode/settings.json`:

```json
{
  // Editor
  "editor.tabSize": 2,
  "editor.insertSpaces": true,
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit"
  },

  // Files
  "files.exclude": {
    "**/node_modules": true,
    "**/dist": true,
    "**/.git": true
  },
  "files.watcherExclude": {
    "**/node_modules/**": true,
    "**/dist/**": true
  },

  // TypeScript/JavaScript
  "typescript.tsdk": "node_modules/typescript/lib",
  "typescript.enablePromptUseWorkspaceTsdk": true,
  "js/ts.implicitProjectConfig.checkJS": true,

  // Emmet (HTML shortcuts)
  "emmet.includeLanguages": {
    "javascript": "javascriptreact",
    "typescript": "typescriptreact"
  },

  // Exclude from search
  "search.exclude": {
    "**/node_modules": true,
    "**/dist": true,
    "**/*.lock": true
  }
}
```

### VS Code Tasks

Create `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Dev Server",
      "type": "shell",
      "command": "npm run dev",
      "isBackground": true,
      "problemMatcher": [],
      "presentation": {
        "reveal": "dedicated",
        "group": "dev"
      }
    },
    {
      "label": "Build",
      "type": "shell",
      "command": "npm run build",
      "problemMatcher": [],
      "presentation": {
        "reveal": "always",
        "group": "build"
      }
    },
    {
      "label": "Preview Build",
      "type": "shell",
      "command": "npm run preview",
      "isBackground": true,
      "problemMatcher": [],
      "presentation": {
        "reveal": "dedicated",
        "group": "build"
      }
    },
    {
      "label": "Clean",
      "type": "shell",
      "command": "npm run clean",
      "problemMatcher": []
    },
    {
      "label": "Type Check",
      "type": "shell",
      "command": "npm run type-check",
      "problemMatcher": "$tsc"
    }
  ]
}
```

### VS Code Launch Configuration

Create `.vscode/launch.json` for debugging:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Launch Chrome",
      "type": "chrome",
      "request": "launch",
      "url": "http://localhost:3000",
      "webRoot": "${workspaceFolder}",
      "breakOnLoad": true,
      "sourceMaps": true
    },
    {
      "name": "Attach to Chrome",
      "type": "chrome",
      "request": "attach",
      "port": 9222,
      "webRoot": "${workspaceFolder}",
      "sourceMaps": true
    }
  ]
}
```

---

## Environment Configuration

### Environment Files

```bash
# .env (committed, defaults)
VITE_APP_TITLE=Shadow Engine
VITE_APP_VERSION=1.0.0

# .env.local (not committed, your overrides)
VITE_DEV_MODE=true
VITE_LOG_LEVEL=debug
```

### Git Ignore for Sensitive Files

Add to `.gitignore`:

```gitignore
# Dependencies
node_modules/

# Build output
dist/
build/

# Environment files
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Debug
*.debug.js
*.debug.map

# Temporary
*.tmp
.cache/
```

---

## Browser Setup for Development

### Chrome DevTools Configuration

1. **Open DevTools**: F12 or right-click → Inspect

2. **Enable Useful Panels**:
   - Console (for errors/logs)
   - Network (for asset loading)
   - Performance (for profiling)
   - Memory (for memory leaks)
   - Layers (for rendering)

3. **Enable Experimental Features**:
   - chrome://flags → Enable WebGPU
   - chrome://flags → Enable GPU rasterization

### Firefox Developer Tools

1. **Open DevTools**: F12 or right-click → Inspect

2. **WebGPU Settings**:
   - about:config → set `dom.webgpu.enabled` to true

### Safari Development

1. **Enable Develop Menu**: Safari → Preferences → Advanced → Show Develop menu

2. **WebGPU Support**:
   - Currently limited, check for future updates

---

## Common Development Issues

### Port Already in Use

```bash
# Error: Port 3000 is already in use
# Solution 1: Kill the process using the port
npx kill-port 3000

# Solution 2: Use a different port
vite --port 3001

# Solution 3: Let Vite find an available port
vite --strictPort false
```

### Module Not Found

```bash
# Error: Cannot find module 'xxx'
# Solution: Reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Or use force flag (use sparingly)
npm install --force
```

### CORS Errors

```bash
# Error: CORS policy blocked request
# Solution: Configure Vite proxy in vite.config.js
server: {
  proxy: {
    '/api': 'http://localhost:8080'
  }
}
```

### Hot Module Replacement Not Working

```bash
# Solution 1: Clear Vite cache
npm run clean
npm run dev

# Solution 2: Check firewall/antivirus
# Some security software blocks WebSocket (needed for HMR)

# Solution 3: Try different browser
# Some browsers have stricter WebSocket policies
```

---

## Quick Start Commands

```bash
# Start development server
npm run dev

# Start with debug logging
npm run dev:debug

# Build for production
npm run build

# Preview production build
npm run preview

# Type checking
npm run type-check

# Clean all build artifacts
npm run clean
```

---

## Development Workflow

### Daily Workflow

```bash
# 1. Pull latest changes
git pull

# 2. Install new dependencies (if package.json changed)
npm install

# 3. Start dev server
npm run dev

# 4. Open browser to localhost:3000

# 5. Make changes, save, browser auto-refreshes

# 6. Test and iterate

# 7. When done, commit changes
git add .
git commit -m "Description of changes"
```

### Before Committing

```bash
# 1. Type check
npm run type-check

# 2. Build test (catches build errors)
npm run build

# 3. Preview build (verify production works)
npm run preview

# 4. Then commit
git commit
```

---

## Related Systems

- [Vite Configuration](./vite-configuration.md) - Build tool settings
- [Build Process](./build-process.md) - Creating production builds
- [WASM Handling](./wasm-handling.md) - WebAssembly setup

---

## Source File Reference

**Setup Files**:
- `package.json` - Dependencies and scripts
- `vite.config.js` - Build configuration
- `.env*` - Environment variables
- `.gitignore` - Excluded files

**Configuration Files**:
- `.vscode/settings.json` - Editor settings
- `.vscode/tasks.json` - Build tasks
- `.vscode/launch.json` - Debug configuration

---

## References

- [Node.js Download](https://nodejs.org/) - JavaScript runtime
- [VS Code](https://code.visualstudio.com/) - Code editor
- [Vite Guide](https://vitejs.dev/guide/) - Build tool guide
- [Git Documentation](https://git-scm.com/doc) - Version control

*Documentation last updated: January 12, 2026*
