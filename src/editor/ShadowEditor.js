/**
 * ShadowEditor.js - GAME EDITOR FOR SHADOW CZAR ENGINE
 * =============================================================================
 *
 * ROLE: A Unity/Blender-inspired game editor built on top of the Shadow Engine.
 * Provides scene editing, asset browsing, object creation, and scene serialization.
 *
 * FEATURES:
 * - Asset Browser: Browse and drag-drop models, splats, audio into scene
 * - Scene Graph: Hierarchical object outliner with parent/child relationships
 * - Inspector: Edit transform, materials, components
 * - Object Tools: Create primitives, lights, duplicate, delete
 * - Scene Save/Load: Serialize scenes to JSON for version control
 * - Play/Edit Mode: Toggle between editing and playing
 * - Undo/Redo: Full command history
 *
 * USAGE:
 *   const editor = new ShadowEditor({
 *     sceneManager,
 *     gizmoManager,
 *     camera,
 *     renderer,
 *     characterController
 *   });
 *
 *   // Open editor
 *   editor.open();
 *
 *   // Save scene
 *   editor.saveScene('my_scene.json');
 *
 * KEYBOARD SHORTCUTS:
 * - Ctrl+E: Toggle Editor
 * - Ctrl+S: Save Scene
 * - Ctrl+Z: Undo
 * - Ctrl+Y: Redo
 * - Ctrl+D: Duplicate Selected
 * - Delete: Remove Selected
 * - F: Focus on Selected
 *
 * =============================================================================
 */

import * as THREE from "three";
import { Logger } from "../utils/logger.js";

// Asset categories
const ASSET_CATEGORIES = {
  models: { name: "Models", icon: "📦", extensions: [".glb", ".gltf"], path: "/models/" },
  splats: { name: "Splats", icon: "🌐", extensions: [".sog"], path: "/splats/" },
  audio: { name: "Audio", icon: "🎵", extensions: [".mp3", ".wav", ".ogg"], path: "/audio/" },
  images: { name: "Images", icon: "🖼️", extensions: [".png", ".jpg", ".webp"], path: "/images/" },
  videos: { name: "Videos", icon: "🎬", extensions: [".webm", ".mp4"], path: "/video/" }
};

// Primitive shapes
const PRIMITIVES = {
  box: { name: "Cube", icon: "📦", create: () => new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial()) },
  sphere: { name: "Sphere", icon: "⚪", create: () => new THREE.Mesh(new THREE.SphereGeometry(0.5, 32, 32), new THREE.MeshStandardMaterial()) },
  cylinder: { name: "Cylinder", icon: "🛢️", create: () => new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 1, 32), new THREE.MeshStandardMaterial()) },
  cone: { name: "Cone", icon: "🔺", create: () => new THREE.Mesh(new THREE.ConeGeometry(0.5, 1, 32), new THREE.MeshStandardMaterial()) },
  plane: { name: "Plane", icon: "⬜", create: () => new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshStandardMaterial({ side: THREE.DoubleSide })) },
  torus: { name: "Torus", icon: "⭕", create: () => new THREE.Mesh(new THREE.TorusGeometry(0.5, 0.2, 16, 100), new THREE.MeshStandardMaterial()) }
};

// Light types
const LIGHT_TYPES = {
  directional: { name: "Directional", icon: "☀️", create: () => new THREE.DirectionalLight(0xffffff, 1) },
  point: { name: "Point", icon: "💡", create: () => new THREE.PointLight(0xffffff, 1, 10) },
  spot: { name: "Spot", icon: "🔦", create: () => new THREE.SpotLight(0xffffff, 1) },
  ambient: { name: "Ambient", icon: "🌐", create: () => new THREE.AmbientLight(0xffffff, 0.5) },
  hemisphere: { name: "Hemisphere", icon: "🌗", create: () => new THREE.HemisphereLight(0xffffff, 0x444444, 0.5) }
};

/**
 * Command base class for undo/redo
 */
class Command {
  execute() { throw new Error("Must implement execute()"); }
  undo() { throw new Error("Must implement undo()"); }
  redo() { throw new Error("Must implement redo()"); }
}

/**
 * CreateObjectCommand
 */
class CreateObjectCommand extends Command {
  constructor(editor, type, options = {}) {
    super();
    this.editor = editor;
    this.type = type;
    this.options = options;
    this.objectId = null;
  }

  execute() {
    const result = this.editor.createObject(this.type, this.options);
    this.objectId = result.id;
    return result;
  }

  undo() {
    if (this.objectId) {
      this.editor.deleteObject(this.objectId);
    }
  }

  redo() {
    if (this.objectId) {
      // Recreate with same ID
      this.options.id = this.objectId;
      this.editor.createObject(this.type, this.options);
    }
  }
}

/**
 * DeleteObjectCommand
 */
class DeleteObjectCommand extends Command {
  constructor(editor, objectId) {
    super();
    this.editor = editor;
    this.objectId = objectId;
    this.savedData = null;
  }

  execute() {
    this.savedData = this.editor.saveObjectData(this.objectId);
    this.editor.deleteObject(this.objectId);
  }

  undo() {
    if (this.savedData) {
      this.editor.restoreObjectData(this.savedData);
    }
  }

  redo() {
    if (this.objectId) {
      this.editor.deleteObject(this.objectId);
    }
  }
}

/**
 * TransformCommand
 */
class TransformCommand extends Command {
  constructor(editor, objectId, oldTransform, newTransform) {
    super();
    this.editor = editor;
    this.objectId = objectId;
    this.oldTransform = oldTransform;
    this.newTransform = newTransform;
  }

  execute() {
    const obj = this.editor.getObjectById(this.objectId);
    if (obj && obj.object) {
      this.applyTransform(obj.object, this.newTransform);
    }
  }

  undo() {
    const obj = this.editor.getObjectById(this.objectId);
    if (obj && obj.object) {
      this.applyTransform(obj.object, this.oldTransform);
    }
  }

  redo() {
    this.execute();
  }

  applyTransform(obj, transform) {
    if (transform.position) obj.position.set(transform.position.x, transform.position.y, transform.position.z);
    if (transform.rotation) obj.rotation.set(transform.rotation.x, transform.rotation.y, transform.rotation.z);
    if (transform.scale) obj.scale.set(transform.scale.x, transform.scale.y, transform.scale.z);
  }
}

/**
 * PropertyChangeCommand
 */
class PropertyChangeCommand extends Command {
  constructor(editor, objectId, property, oldValue, newValue) {
    super();
    this.editor = editor;
    this.objectId = objectId;
    this.property = property;
    this.oldValue = oldValue;
    this.newValue = newValue;
  }

  execute() {
    this.applyValue(this.newValue);
  }

  undo() {
    this.applyValue(this.oldValue);
  }

  redo() {
    this.execute();
  }

  applyValue(value) {
    const obj = this.editor.getObjectById(this.objectId);
    if (!obj) return;

    const parts = this.property.split(".");
    let target = obj.object;

    for (let i = 0; i < parts.length - 1; i++) {
      target = target[parts[i]];
    }

    target[parts[parts.length - 1]] = value;
  }
}

class ShadowEditor {
  constructor(options = {}) {
    this.sceneManager = options.sceneManager || null;
    this.gizmoManager = options.gizmoManager || null;
    this.camera = options.camera || null;
    this.renderer = options.renderer || null;
    this.characterController = options.characterController || null;
    this.lightManager = options.lightManager || window.lightManager || null;

    this.logger = new Logger("ShadowEditor", true);

    // Editor state
    this.isOpen = false;
    this.isPlaying = false;
    this.selectedObjectId = null;
    this.expandedObjects = new Set();
    this.searchQuery = "";
    this.filterType = "all";

    // Scene data for serialization
    this.sceneObjects = new Map(); // id -> { object, data, children: [] }
    this.objectParentMap = new Map(); // childId -> parentId
    this.nextObjectId = 1;

    // Undo/Redo
    this.undoStack = [];
    this.redoStack = [];

    // Asset cache
    this.assetCache = null;

    // Build UI
    this.createEditorHTML();
    this.injectStyles();
    this.bindEvents();

    // Keyboard shortcuts
    this.bindKeyboardShortcuts();

    // Global access
    window.shadowEditor = this;

    // Initialize scene data
    this.refreshSceneData();

    this.logger.log("ShadowEditor initialized");
  }

  /**
   * Create editor HTML structure
   */
  createEditorHTML() {
    const editor = document.createElement("div");
    editor.id = "shadow-editor";
    editor.className = "shadow-editor hidden";

    editor.innerHTML = `
      <div class="se-overlay"></div>

      <div class="se-container">
        <!-- Top Toolbar -->
        <div class="se-toolbar">
          <div class="se-title">
            <span class="se-icon">🎮</span>
            <span>SHADOW EDITOR</span>
            <span class="se-scene-name" id="se-scene-name">Untitled Scene</span>
          </div>

          <div class="se-tools">
            <button class="se-tool-btn" data-tool="select" title="Select (Q)">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <path d="M4 4l16 8-7 2-2 7z"/>
              </svg>
            </button>
            <button class="se-tool-btn" data-tool="move" title="Move (G)">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M5 9l-3 3 3 3M9 5l3-3 3 3M19 9l3 3-3 3M15 19l-3 3-3-3M2 12h20M12 2v20"/>
              </svg>
            </button>
            <button class="se-tool-btn" data-tool="rotate" title="Rotate (R)">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M21 12a9 9 0 11-9-9"/>
                <path d="M21 3v6h-6"/>
              </svg>
            </button>
            <button class="se-tool-btn" data-tool="scale" title="Scale (S)">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M21 3L3 21M3 3l18 18M8 3v18M16 3v18M3 8h18M3 16h18"/>
              </svg>
            </button>
          </div>

          <div class="se-playback">
            <button class="se-play-btn" id="se-play-btn" title="Play (Ctrl+P)">
              <span class="se-play-icon">▶</span>
              <span class="se-play-text">PLAY</span>
            </button>
          </div>

          <div class="se-actions">
            <button class="se-action-btn" id="se-save-btn" title="Save Scene (Ctrl+S)">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
                <path d="M17 21v-8H7v8M7 3v5h8"/>
              </svg>
              Save
            </button>
            <button class="se-action-btn" id="se-load-btn" title="Load Scene (Ctrl+O)">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>
              </svg>
              Load
            </button>
            <button class="se-action-btn" id="se-export-btn" title="Export to sceneData.js">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/>
              </svg>
              Export
            </button>
            <button class="se-close-btn" id="se-close-btn" title="Close (Ctrl+Shift+E)">×</button>
          </div>
        </div>

        <!-- Main Content -->
        <div class="se-content">
          <!-- Left Panel: Asset Browser -->
          <div class="se-panel se-assets-panel">
            <div class="se-panel-header">
              <h3>Assets</h3>
              <button class="se-refresh-btn" id="se-refresh-assets" title="Refresh">↻</button>
            </div>
            <div class="se-asset-tabs" id="se-asset-tabs"></div>
            <div class="se-asset-list" id="se-asset-list">
              <div class="se-empty">Loading assets...</div>
            </div>
            <div class="se-create-panel">
              <div class="se-panel-header">
                <h3>Create</h3>
              </div>
              <div class="se-primitives-grid" id="se-primitives"></div>
              <div class="se-lights-grid" id="se-lights"></div>
            </div>
          </div>

          <!-- Center: Scene Graph -->
          <div class="se-panel se-scene-panel">
            <div class="se-panel-header">
              <h3>Scene Graph</h3>
              <div class="se-stats" id="se-stats">0 objects</div>
            </div>
            <div class="se-scene-toolbar">
              <input type="text" id="se-search" placeholder="Search objects..." />
              <select id="se-filter">
                <option value="all">All</option>
                <option value="gltf">Models</option>
                <option value="splat">Splats</option>
                <option value="light">Lights</option>
                <option value="collider">Colliders</option>
              </select>
            </div>
            <div class="se-scene-graph" id="se-scene-graph">
              <div class="se-empty">No objects in scene</div>
            </div>
          </div>

          <!-- Right Panel: Inspector -->
          <div class="se-panel se-inspector-panel">
            <div class="se-panel-header">
              <h3>Inspector</h3>
            </div>
            <div class="se-inspector" id="se-inspector">
              <div class="se-empty">Select an object to inspect</div>
            </div>
          </div>
        </div>

        <!-- Bottom: Status Bar -->
        <div class="se-status-bar">
          <span id="se-status">Ready</span>
          <span class="se-undo-redo">
            <button id="se-undo-btn" title="Undo (Ctrl+Z)" disabled>↶ Undo</button>
            <button id="se-redo-btn" title="Redo (Ctrl+Y)" disabled>↷ Redo</button>
          </span>
          <span class="se-shortcuts">Ctrl+E: Editor | Ctrl+S: Save | Del: Delete</span>
        </div>
      </div>
    `;

    document.body.appendChild(editor);
    this.element = editor;
  }

  /**
   * Inject CSS styles
   */
  injectStyles() {
    const styleId = "shadow-editor-styles";
    if (document.getElementById(styleId)) return;

    const style = document.createElement("style");
    style.id = styleId;
    style.textContent = `
      /* ===== Shadow Editor Styles ===== */
      .shadow-editor {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 10000;
        pointer-events: none;
        font-family: 'Segoe UI', system-ui, sans-serif;
        font-size: 12px;
        color: #e0e0e0;
      }

      .shadow-editor.hidden {
        display: none;
      }

      .shadow-editor:not(.hidden) {
        display: block;
      }

      .se-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.6);
        pointer-events: auto;
      }

      .se-container {
        position: absolute;
        top: 10px;
        left: 10px;
        right: 10px;
        bottom: 10px;
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
        border-radius: 12px;
        box-shadow: 0 25px 80px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(255, 255, 255, 0.1);
        display: flex;
        flex-direction: column;
        overflow: hidden;
        pointer-events: auto;
      }

      /* Toolbar */
      .se-toolbar {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 16px;
        background: rgba(0, 0, 0, 0.4);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      }

      .se-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 700;
        font-size: 14px;
        color: #4a9eff;
      }

      .se-icon {
        font-size: 20px;
      }

      .se-scene-name {
        margin-left: 12px;
        padding: 4px 10px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 6px;
        font-size: 11px;
        color: #888;
      }

      .se-tools {
        display: flex;
        gap: 4px;
        padding: 0 16px;
        border-left: 1px solid rgba(255, 255, 255, 0.1);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
      }

      .se-tool-btn {
        width: 36px;
        height: 32px;
        border: none;
        background: transparent;
        color: #888;
        border-radius: 6px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
      }

      .se-tool-btn:hover {
        background: rgba(255, 255, 255, 0.1);
        color: #ccc;
      }

      .se-tool-btn.active {
        background: rgba(74, 158, 255, 0.3);
        color: #4a9eff;
      }

      .se-playback {
        margin-left: auto;
      }

      .se-play-btn {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 20px;
        background: #2ecc71;
        color: #000;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.2s;
      }

      .se-play-btn:hover {
        background: #27ae60;
        transform: scale(1.02);
      }

      .se-play-btn.playing {
        background: #e74c3c;
      }

      .se-play-btn.playing:hover {
        background: #c0392b;
      }

      .se-actions {
        display: flex;
        gap: 8px;
        margin-left: 12px;
      }

      .se-action-btn {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 8px 14px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #ccc;
        border-radius: 6px;
        cursor: pointer;
        font-size: 11px;
        font-weight: 600;
        transition: all 0.2s;
      }

      .se-action-btn:hover {
        background: rgba(255, 255, 255, 0.15);
        color: #fff;
      }

      .se-close-btn {
        width: 36px;
        padding: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        color: #e74c3c;
      }

      .se-close-btn:hover {
        background: rgba(231, 76, 60, 0.2);
      }

      /* Content Area */
      .se-content {
        display: flex;
        flex: 1;
        overflow: hidden;
      }

      .se-panel {
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }

      .se-assets-panel {
        width: 260px;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
      }

      .se-scene-panel {
        flex: 1;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        min-width: 0;
      }

      .se-inspector-panel {
        width: 320px;
      }

      /* Panel Headers */
      .se-panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 14px;
        background: rgba(0, 0, 0, 0.3);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      }

      .se-panel-header h3 {
        margin: 0;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #666;
        font-weight: 700;
      }

      .se-stats {
        font-size: 10px;
        color: #666;
        background: rgba(0, 0, 0, 0.3);
        padding: 3px 8px;
        border-radius: 10px;
      }

      /* Asset Browser */
      .se-asset-tabs {
        display: flex;
        padding: 8px;
        gap: 4px;
        overflow-x: auto;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      }

      .se-asset-tab {
        padding: 6px 12px;
        background: transparent;
        border: none;
        color: #888;
        border-radius: 6px;
        cursor: pointer;
        font-size: 11px;
        white-space: nowrap;
        transition: all 0.2s;
      }

      .se-asset-tab:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #ccc;
      }

      .se-asset-tab.active {
        background: rgba(74, 158, 255, 0.2);
        color: #4a9eff;
      }

      .se-asset-list {
        flex: 1;
        overflow-y: auto;
        padding: 8px;
      }

      .se-asset-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 10px;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.15s;
      }

      .se-asset-item:hover {
        background: rgba(255, 255, 255, 0.08);
      }

      .se-asset-item .se-asset-icon {
        font-size: 24px;
      }

      .se-asset-item .se-asset-info {
        flex: 1;
        min-width: 0;
      }

      .se-asset-item .se-asset-name {
        font-size: 11px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .se-asset-item .se-asset-path {
        font-size: 9px;
        color: #666;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .se-asset-item .se-add-btn {
        width: 24px;
        height: 24px;
        border: none;
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border-radius: 4px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        opacity: 0;
        transition: all 0.2s;
      }

      .se-asset-item:hover .se-add-btn {
        opacity: 1;
      }

      .se-asset-item .se-add-btn:hover {
        background: #2ecc71;
        color: #000;
      }

      /* Create Panel */
      .se-create-panel {
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        max-height: 280px;
        overflow: hidden;
      }

      .se-primitives-grid, .se-lights-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 6px;
        padding: 10px;
      }

      .se-primitive-btn, .se-light-btn {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        padding: 12px 8px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s;
      }

      .se-primitive-btn:hover, .se-light-btn:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.15);
      }

      .se-primitive-btn span, .se-light-btn span {
        font-size: 24px;
      }

      .se-primitive-btn .se-name, .se-light-btn .se-name {
        font-size: 9px;
        color: #888;
        text-transform: uppercase;
      }

      /* Scene Toolbar */
      .se-scene-toolbar {
        display: flex;
        gap: 8px;
        padding: 10px 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      }

      .se-scene-toolbar input {
        flex: 1;
        padding: 7px 12px;
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        color: #e0e0e0;
        font-size: 11px;
      }

      .se-scene-toolbar input:focus {
        outline: none;
        border-color: #4a9eff;
      }

      .se-scene-toolbar select {
        padding: 7px 10px;
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        color: #e0e0e0;
        font-size: 11px;
        cursor: pointer;
      }

      /* Scene Graph */
      .se-scene-graph {
        flex: 1;
        overflow-y: auto;
        padding: 8px 0;
      }

      .se-scene-graph::-webkit-scrollbar {
        width: 8px;
      }

      .se-scene-graph::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
      }

      .se-scene-graph::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 4px;
      }

      .se-empty {
        padding: 60px 20px;
        text-align: center;
        color: #555;
        font-style: italic;
      }

      /* Scene Graph Items */
      .se-scene-item {
        display: flex;
        align-items: center;
        padding: 6px 14px;
        cursor: pointer;
        user-select: none;
        transition: all 0.15s;
        border-left: 3px solid transparent;
      }

      .se-scene-item:hover {
        background: rgba(255, 255, 255, 0.04);
      }

      .se-scene-item.selected {
        background: rgba(74, 158, 255, 0.15);
        border-left-color: #4a9eff;
      }

      .se-scene-item .se-expand-btn {
        width: 18px;
        height: 18px;
        border: none;
        background: none;
        color: #666;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 4px;
        font-size: 10px;
      }

      .se-scene-item .se-expand-btn.expanded {
        transform: rotate(90deg);
      }

      .se-scene-item .se-item-icon {
        width: 18px;
        font-size: 14px;
        text-align: center;
        margin-right: 8px;
      }

      .se-scene-item .se-item-name {
        flex: 1;
        font-size: 11px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .se-scene-item .se-item-type {
        font-size: 8px;
        color: #666;
        text-transform: uppercase;
        padding: 2px 5px;
        border-radius: 3px;
        background: rgba(0, 0, 0, 0.3);
      }

      .se-scene-item .se-item-visible {
        width: 20px;
        height: 20px;
        border: none;
        background: none;
        color: #666;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-left: 6px;
      }

      .se-scene-item .se-item-visible.hidden {
        color: #e74c3c;
      }

      /* Scene Children */
      .se-scene-children {
        display: none;
        background: rgba(0, 0, 0, 0.2);
      }

      .se-scene-children.visible {
        display: block;
      }

      .se-child-item {
        display: flex;
        align-items: center;
        padding: 4px 14px 4px 48px;
        cursor: pointer;
        font-size: 10px;
        color: #888;
      }

      .se-child-item:hover {
        background: rgba(255, 255, 255, 0.04);
        color: #ccc;
      }

      .se-child-item .se-child-icon {
        margin-right: 6px;
      }

      /* Inspector */
      .se-inspector {
        flex: 1;
        overflow-y: auto;
        padding: 12px;
      }

      .se-inspector::-webkit-scrollbar {
        width: 6px;
      }

      .se-inspector::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
      }

      .se-prop-section {
        margin-bottom: 16px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        overflow: hidden;
      }

      .se-prop-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 14px;
        background: rgba(255, 255, 255, 0.05);
        cursor: pointer;
        user-select: none;
      }

      .se-prop-header:hover {
        background: rgba(255, 255, 255, 0.08);
      }

      .se-prop-header .se-prop-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      .se-prop-header .se-prop-toggle {
        width: 18px;
        height: 18px;
        border: none;
        background: none;
        color: #888;
        cursor: pointer;
        font-size: 10px;
      }

      .se-prop-content {
        display: none;
        padding: 12px;
      }

      .se-prop-content.visible {
        display: block;
      }

      .se-prop-row {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
      }

      .se-prop-row:last-child {
        margin-bottom: 0;
      }

      .se-prop-label {
        width: 70px;
        font-size: 10px;
        color: #888;
        text-transform: uppercase;
      }

      .se-prop-inputs {
        flex: 1;
        display: flex;
        gap: 6px;
      }

      .se-prop-input-group {
        flex: 1;
        display: flex;
        align-items: center;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 6px;
        overflow: hidden;
      }

      .se-prop-input-group .se-axis {
        width: 18px;
        font-size: 9px;
        text-align: center;
        color: #666;
        background: rgba(0, 0, 0, 0.5);
      }

      .se-prop-input-group.x .se-axis { color: #e74c3c; }
      .se-prop-input-group.y .se-axis { color: #2ecc71; }
      .se-prop-input-group.z .se-axis { color: #3498db; }

      .se-prop-input {
        width: 100%;
        padding: 6px 8px;
        background: transparent;
        border: none;
        color: #e0e0e0;
        font-size: 11px;
        font-family: 'Consolas', monospace;
      }

      .se-prop-input:focus {
        outline: none;
        background: rgba(74, 158, 255, 0.15);
      }

      .se-prop-value {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
      }

      .se-prop-value:last-child {
        border-bottom: none;
      }

      .se-prop-value-label {
        font-size: 10px;
        color: #888;
      }

      .se-prop-value-value {
        font-family: 'Consolas', monospace;
        font-size: 10px;
        color: #e0e0e0;
      }

      .se-action-buttons {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-top: 12px;
      }

      .se-action-button {
        padding: 10px;
        border: none;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 600;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        transition: all 0.2s;
      }

      .se-action-button.primary {
        background: #4a9eff;
        color: #fff;
      }

      .se-action-button.primary:hover {
        background: #3a8eef;
      }

      .se-action-button.danger {
        background: #e74c3c;
        color: #fff;
      }

      .se-action-button.danger:hover {
        background: #c0392b;
      }

      .se-action-button.secondary {
        background: rgba(255, 255, 255, 0.08);
        color: #ccc;
      }

      .se-action-button.secondary:hover {
        background: rgba(255, 255, 255, 0.15);
        color: #fff;
      }

      /* Status Bar */
      .se-status-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 16px;
        background: rgba(0, 0, 0, 0.3);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 10px;
        color: #666;
      }

      .se-undo-redo {
        display: flex;
        gap: 8px;
      }

      .se-undo-redo button {
        padding: 4px 10px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        color: #888;
        cursor: pointer;
        font-size: 10px;
      }

      .se-undo-redo button:hover:not(:disabled) {
        background: rgba(255, 255, 255, 0.1);
        color: #ccc;
      }

      .se-undo-redo button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }

      .se-shortcuts {
        color: #666;
      }

      /* Refresh button */
      .se-refresh-btn {
        width: 24px;
        height: 24px;
        border: none;
        background: transparent;
        color: #888;
        cursor: pointer;
        border-radius: 4px;
        font-size: 14px;
      }

      .se-refresh-btn:hover {
        background: rgba(255, 255, 255, 0.1);
        color: #ccc;
      }

      /* Type colors */
      .type-gltf { color: #3498db; }
      .type-splat { color: #e74c3c; }
      .type-light { color: #f1c40f; }
      .type-video { color: #9b59b6; }
      .type-collider { color: #1abc9c; }
      .type-primitive { color: #e91e63; }
    `;

    document.head.appendChild(style);
  }

  /**
   * Bind event listeners
   */
  bindEvents() {
    // Close
    this.element.querySelector("#se-close-btn").addEventListener("click", () => this.close());
    this.element.querySelector(".se-overlay").addEventListener("click", () => this.close());

    // Save/Load/Export
    this.element.querySelector("#se-save-btn").addEventListener("click", () => this.showSaveDialog());
    this.element.querySelector("#se-load-btn").addEventListener("click", () => this.showLoadDialog());
    this.element.querySelector("#se-export-btn").addEventListener("click", () => this.exportToCode());

    // Play toggle
    this.element.querySelector("#se-play-btn").addEventListener("click", () => this.togglePlayMode());

    // Undo/Redo
    this.element.querySelector("#se-undo-btn").addEventListener("click", () => this.undo());
    this.element.querySelector("#se-redo-btn").addEventListener("click", () => this.redo());

    // Search
    this.element.querySelector("#se-search").addEventListener("input", (e) => {
      this.searchQuery = e.target.value.toLowerCase();
      this.refreshSceneGraph();
    });

    // Filter
    this.element.querySelector("#se-filter").addEventListener("change", (e) => {
      this.filterType = e.target.value;
      this.refreshSceneGraph();
    });

    // Refresh assets
    this.element.querySelector("#se-refresh-assets").addEventListener("click", () => {
      this.assetCache = null;
      this.refreshAssetBrowser();
    });

    // Double-click to select
    if (this.renderer && this.camera) {
      this.renderer.domElement.addEventListener("dblclick", (e) => this.handleDoubleClick(e));
    }

    // Build asset tabs
    this.buildAssetTabs();

    // Build primitives and lights
    this.buildCreatePanel();
  }

  /**
   * Build asset category tabs
   */
  buildAssetTabs() {
    const tabsContainer = document.getElementById("se-asset-tabs");

    Object.entries(ASSET_CATEGORIES).forEach(([key, category], index) => {
      const tab = document.createElement("button");
      tab.className = `se-asset-tab ${index === 0 ? "active" : ""}`;
      tab.dataset.category = key;
      tab.textContent = `${category.icon} ${category.name}`;
      tab.addEventListener("click", () => {
        document.querySelectorAll(".se-asset-tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        this.refreshAssetBrowser(key);
      });
      tabsContainer.appendChild(tab);
    });

    // Load first category
    this.refreshAssetBrowser("models");
  }

  /**
   * Build primitives and lights grid
   */
  buildCreatePanel() {
    const primitivesContainer = document.getElementById("se-primitives");
    const lightsContainer = document.getElementById("se-lights");

    // Primitives
    Object.entries(PRIMITIVES).forEach(([key, primitive]) => {
      const btn = document.createElement("div");
      btn.className = "se-primitive-btn";
      btn.innerHTML = `<span>${primitive.icon}</span><span class="se-name">${primitive.name}</span>`;
      btn.addEventListener("click", () => this.executeCommand(new CreateObjectCommand(this, "primitive", { shape: key })));
      primitivesContainer.appendChild(btn);
    });

    // Lights
    Object.entries(LIGHT_TYPES).forEach(([key, light]) => {
      const btn = document.createElement("div");
      btn.className = "se-light-btn";
      btn.innerHTML = `<span>${light.icon}</span><span class="se-name">${light.name}</span>`;
      btn.addEventListener("click", () => this.executeCommand(new CreateObjectCommand(this, "light", { type: key })));
      lightsContainer.appendChild(btn);
    });
  }

  /**
   * Refresh asset browser for a category
   */
  async refreshAssetBrowser(category = "models") {
    const listContainer = document.getElementById("se-asset-list");

    if (!this.assetCache) {
      this.assetCache = {};
      // Could fetch from server in a real app
      // For now, use known assets
      this.assetCache.models = [
        { name: "chair.glb", path: "/models/chair.glb" },
        { name: "table.glb", path: "/models/table.glb" },
        { name: "crate.glb", path: "/models/crate.glb" },
        { name: "lamp.glb", path: "/models/lamp.glb" },
      ];
      this.assetCache.splats = [
        { name: "plaza.sog", path: "/splats/plaza_16m.sog" },
        { name: "fourWay.sog", path: "/splats/FourWay.sog" },
        { name: "alleyIntro.sog", path: "/splats/AlleyIntro.sog" },
      ];
      this.assetCache.audio = [
        { name: "music_theme.mp3", path: "/audio/music/office.mp3" },
        { name: "sfx_door.mp3", path: "/audio/sfx/door.mp3" },
      ];
    }

    const assets = this.assetCache[category] || [];
    const categoryInfo = ASSET_CATEGORIES[category];

    listContainer.innerHTML = "";

    if (assets.length === 0) {
      listContainer.innerHTML = `<div class="se-empty">No ${categoryInfo.name} found</div>`;
      return;
    }

    assets.forEach(asset => {
      const item = document.createElement("div");
      item.className = "se-asset-item";
      item.innerHTML = `
        <span class="se-asset-icon">${categoryInfo.icon}</span>
        <div class="se-asset-info">
          <div class="se-asset-name">${asset.name}</div>
          <div class="se-asset-path">${asset.path}</div>
        </div>
        <button class="se-add-btn" title="Add to scene">+</button>
      `;

      item.querySelector(".se-add-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        this.executeCommand(new CreateObjectCommand(this, category, asset));
      });

      item.addEventListener("dblclick", () => {
        this.executeCommand(new CreateObjectCommand(this, category, asset));
      });

      listContainer.appendChild(item);
    });
  }

  /**
   * Bind keyboard shortcuts
   */
  bindKeyboardShortcuts() {
    this.keydownHandler = (e) => {
      if (document.activeElement.tagName === "INPUT" ||
          document.activeElement.tagName === "TEXTAREA") {
        return;
      }

      // Ctrl+E: Toggle editor
      if (e.ctrlKey && (e.key === "e" || e.key === "E")) {
        e.preventDefault();
        this.toggle();
        return;
      }

      if (!this.isOpen) return;

      // Ctrl+S: Save
      if (e.ctrlKey && (e.key === "s" || e.key === "S")) {
        e.preventDefault();
        this.showSaveDialog();
        return;
      }

      // Ctrl+O: Load
      if (e.ctrlKey && (e.key === "o" || e.key === "O")) {
        e.preventDefault();
        this.showLoadDialog();
        return;
      }

      // Ctrl+Z: Undo
      if (e.ctrlKey && (e.key === "z" || e.key === "Z") && !e.shiftKey) {
        e.preventDefault();
        this.undo();
        return;
      }

      // Ctrl+Y or Ctrl+Shift+Z: Redo
      if ((e.ctrlKey && (e.key === "y" || e.key === "Y")) ||
          (e.ctrlKey && e.shiftKey && (e.key === "z" || e.key === "Z"))) {
        e.preventDefault();
        this.redo();
        return;
      }

      // Ctrl+D: Duplicate
      if (e.ctrlKey && (e.key === "d" || e.key === "D")) {
        e.preventDefault();
        this.duplicateSelected();
        return;
      }

      // Delete
      if (e.key === "Delete") {
        this.deleteSelected();
        return;
      }

      // F: Focus
      if (e.key === "f" || e.key === "F") {
        this.focusOnSelected();
        return;
      }

      // Escape: Deselect
      if (e.key === "Escape") {
        this.deselectObject();
        return;
      }

      // Tools
      if (e.key === "q" || e.key === "Q") this.setTool("select");
      if (e.key === "g" || e.key === "G") this.setTool("move");
      if (e.key === "r" || e.key === "R") this.setTool("rotate");
      if (e.key === "s" || e.key === "S" && !e.ctrlKey) this.setTool("scale");
    };

    window.addEventListener("keydown", this.keydownHandler);
  }

  /**
   * Handle double-click on canvas
   */
  handleDoubleClick(event) {
    if (!this.sceneManager) return;

    const rect = this.renderer.domElement.getBoundingClientRect();
    const mouse = {
      x: ((event.clientX - rect.left) / rect.width) * 2 - 1,
      y: -((event.clientY - rect.top) / rect.height) * 2 + 1
    };

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, this.camera);

    const objects = [];
    this.sceneManager.objects.forEach((obj) => {
      if (obj.isObject3D) {
        objects.push(obj);
        obj.traverse((child) => {
          if (child.isObject3D) objects.push(child);
        });
      }
    });

    const intersects = raycaster.intersectObjects(objects, false);

    if (intersects.length > 0) {
      let obj = intersects[0].object;
      while (obj.parent && obj.parent.type !== "Scene") {
        obj = obj.parent;
      }

      for (const [id, o] of this.sceneManager.objects) {
        if (o === obj) {
          this.selectObject(id);
          this.open();
          return;
        }
      }
    }
  }

  /**
   * Refresh scene data from managers
   */
  refreshSceneData() {
    this.sceneObjects.clear();

    // From sceneManager
    if (this.sceneManager) {
      this.sceneManager.objects.forEach((obj, id) => {
        const data = this.sceneManager.objectData.get(id) || {};
        this.sceneObjects.set(id, {
          id,
          object: obj,
          data,
          type: data.type || "other",
          source: "scene"
        });
      });
    }

    // From lightManager
    if (this.lightManager?.lights) {
      this.lightManager.lights.forEach((light, id) => {
        this.sceneObjects.set(id, {
          id,
          object: light,
          data: { type: "light" },
          type: "light",
          source: "light"
        });
      });
    }

    // From gizmoManager (spawned objects)
    if (this.gizmoManager?.objects) {
      this.gizmoManager.objects.forEach((item) => {
        if (!this.sceneObjects.has(item.id)) {
          this.sceneObjects.set(item.id, {
            id: item.id,
            object: item.object,
            data: { type: item.type },
            type: item.type,
            source: "gizmo"
          });
        }
      });
    }

    this.refreshSceneGraph();
    this.updateStats();
  }

  /**
   * Refresh the scene graph UI
   */
  refreshSceneGraph() {
    const graph = document.getElementById("se-scene-graph");
    if (!graph) return;

    graph.innerHTML = "";

    let count = 0;
    const items = [];

    // Filter objects
    this.sceneObjects.forEach((obj) => {
      if (this.filterType !== "all" && obj.type !== this.filterType) return;

      const name = obj.data?.description || obj.id;
      if (this.searchQuery && !name.toLowerCase().includes(this.searchQuery)) return;

      items.push(obj);
      count++;
    });

    if (items.length === 0) {
      graph.innerHTML = `<div class="se-empty">No objects found</div>`;
      return;
    }

    // Sort by type
    items.sort((a, b) => a.type.localeCompare(b.type));

    // Create items
    items.forEach(obj => {
      const el = this.createSceneItemElement(obj);
      graph.appendChild(el);
    });

    this.updateStats();
  }

  /**
   * Create a scene graph item element
   */
  createSceneItemElement(obj) {
    const div = document.createElement("div");
    div.className = "se-scene-item";
    div.dataset.objectId = obj.id;
    if (this.selectedObjectId === obj.id) {
      div.classList.add("selected");
    }

    const isVisible = obj.object.visible !== false;
    const hasChildren = obj.object.children && obj.object.children.length > 0;

    div.innerHTML = `
      <button class="se-expand-btn" style="visibility: ${hasChildren ? "visible" : "hidden"}">▶</button>
      <span class="se-item-icon">${this.getObjectIcon(obj.type)}</span>
      <span class="se-item-name">${obj.data?.description || obj.id}</span>
      <span class="se-item-type type-${obj.type}">${obj.type}</span>
      <button class="se-item-visible ${isVisible ? "" : "hidden"}">${isVisible ? "👁️" : "👁️‍🗨️"}</button>
    `;

    div.addEventListener("click", (e) => {
      if (e.target.closest(".se-expand-btn") || e.target.closest(".se-item-visible")) return;
      this.selectObject(obj.id);
    });

    // Visibility toggle
    const visBtn = div.querySelector(".se-item-visible");
    visBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      obj.object.visible = !obj.object.visible;
      visBtn.classList.toggle("hidden", !obj.object.visible);
      visBtn.textContent = obj.object.visible ? "👁️" : "👁️‍🗨️";
      this.setStatus(`${obj.object.visible ? "Showed" : "Hid"} ${obj.id}`);
    });

    // Expand button
    if (hasChildren) {
      const expandBtn = div.querySelector(".se-expand-btn");
      expandBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.toggleObjectExpansion(obj.id, div, obj.object);
      });
    }

    return div;
  }

  /**
   * Toggle object children expansion
   */
  toggleObjectExpansion(id, element, object) {
    let children = element.querySelector(".se-scene-children");

    if (children) {
      children.classList.toggle("visible");
      element.querySelector(".se-expand-btn").classList.toggle("expanded", children.classList.contains("visible"));
    } else {
      children = document.createElement("div");
      children.className = "se-scene-children visible";

      object.children.forEach((child, index) => {
        const childDiv = document.createElement("div");
        childDiv.className = "se-child-item";

        const icon = child.isMesh ? "🔷" :
                     child.isLight ? "💡" :
                     child.isCamera ? "📷" :
                     child.isGroup ? "📁" : "📄";

        childDiv.innerHTML = `
          <span class="se-child-icon">${icon}</span>
          <span>${child.name || child.type || `child_${index}`}</span>
        `;

        children.appendChild(childDiv);
      });

      element.appendChild(children);
      element.querySelector(".se-expand-btn").classList.add("expanded");
    }
  }

  /**
   * Select an object
   */
  selectObject(id) {
    // Deselect previous
    if (this.selectedObjectId) {
      const prev = this.element.querySelector(`.se-scene-item[data-object-id="${this.selectedObjectId}"]`);
      if (prev) prev.classList.remove("selected");
    }

    this.selectedObjectId = id;

    // Select new
    const current = this.element.querySelector(`.se-scene-item[data-object-id="${id}"]`);
    if (current) current.classList.add("selected");

    this.updateInspector(id);

    // Add gizmo if not exists
    if (this.gizmoManager) {
      const obj = this.getObjectById(id);
      if (obj && !this.gizmoManager.objects.some(o => o.id === id)) {
        this.gizmoManager.registerObject(obj.object, id, obj.data?.type || "object");
      }
      this.gizmoManager.activeObject = { id, object: obj.object, type: obj.data?.type };
    }

    this.setStatus(`Selected: ${id}`);
  }

  /**
   * Deselect current object
   */
  deselectObject() {
    if (this.selectedObjectId) {
      const prev = this.element.querySelector(`.se-scene-item[data-object-id="${this.selectedObjectId}"]`);
      if (prev) prev.classList.remove("selected");
    }

    this.selectedObjectId = null;
    this.updateInspector(null);
  }

  /**
   * Update inspector panel
   */
  updateInspector(id) {
    const inspector = document.getElementById("se-inspector");
    if (!inspector) return;

    if (!id) {
      inspector.innerHTML = `<div class="se-empty">Select an object to inspect</div>`;
      return;
    }

    const obj = this.getObjectById(id);
    if (!obj) {
      inspector.innerHTML = `<div class="se-empty">Object not found</div>`;
      return;
    }

    const o = obj.object;
    const data = obj.data || {};

    inspector.innerHTML = `
      ${this.createTransformSection(id, o, data)}
      ${this.createMaterialSection(id, o, data)}
      ${this.createInfoSection(id, o, data)}
      ${this.createActionsSection(id, o, data)}
    `;

    this.bindInspectorInputs(id, o);
  }

  /**
   * Create transform section
   */
  createTransformSection(id, obj, data) {
    const pos = obj.position;
    const rot = obj.rotation;
    const scale = obj.scale;

    return `
      <div class="se-prop-section">
        <div class="se-prop-header">
          <div class="se-prop-title"><span>📍</span> Transform</div>
          <button class="se-prop-toggle">▼</button>
        </div>
        <div class="se-prop-content visible">
          <div class="se-prop-row">
            <span class="se-prop-label">Position</span>
            <div class="se-prop-inputs">
              <div class="se-prop-input-group x"><span class="se-axis">X</span><input type="number" class="se-prop-input se-pos-x" step="0.1" value="${pos.x.toFixed(3)}" /></div>
              <div class="se-prop-input-group y"><span class="se-axis">Y</span><input type="number" class="se-prop-input se-pos-y" step="0.1" value="${pos.y.toFixed(3)}" /></div>
              <div class="se-prop-input-group z"><span class="se-axis">Z</span><input type="number" class="se-prop-input se-pos-z" step="0.1" value="${pos.z.toFixed(3)}" /></div>
            </div>
          </div>
          <div class="se-prop-row">
            <span class="se-prop-label">Rotation</span>
            <div class="se-prop-inputs">
              <div class="se-prop-input-group x"><span class="se-axis">X</span><input type="number" class="se-prop-input se-rot-x" step="0.01" value="${rot.x.toFixed(4)}" /></div>
              <div class="se-prop-input-group y"><span class="se-axis">Y</span><input type="number" class="se-prop-input se-rot-y" step="0.01" value="${rot.y.toFixed(4)}" /></div>
              <div class="se-prop-input-group z"><span class="se-axis">Z</span><input type="number" class="se-prop-input se-rot-z" step="0.01" value="${rot.z.toFixed(4)}" /></div>
            </div>
          </div>
          <div class="se-prop-row">
            <span class="se-prop-label">Scale</span>
            <div class="se-prop-inputs">
              <div class="se-prop-input-group x"><span class="se-axis">X</span><input type="number" class="se-prop-input se-scale-x" step="0.1" value="${scale.x.toFixed(3)}" /></div>
              <div class="se-prop-input-group y"><span class="se-axis">Y</span><input type="number" class="se-prop-input se-scale-y" step="0.1" value="${scale.y.toFixed(3)}" /></div>
              <div class="se-prop-input-group z"><span class="se-axis">Z</span><input type="number" class="se-prop-input se-scale-z" step="0.1" value="${scale.z.toFixed(3)}" /></div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  /**
   * Create material section
   */
  createMaterialSection(id, obj, data) {
    // Try to get material
    let material = obj.material || (obj.children && obj.children[0]?.material);

    if (!material) {
      return `
        <div class="se-prop-section">
          <div class="se-prop-header">
            <div class="se-prop-title"><span>🎨</span> Material</div>
            <button class="se-prop-toggle">▼</button>
          </div>
          <div class="se-prop-content visible">
            <div class="se-empty" style="padding: 20px;">No material</div>
          </div>
        </div>
      `;
    }

    const color = material.color ? `#${material.color.getHexString()}` : "#ffffff";
    const metalness = material.metalness ?? 0;
    const roughness = material.roughness ?? 0.5;

    return `
      <div class="se-prop-section">
        <div class="se-prop-header">
          <div class="se-prop-title"><span>🎨</span> Material</div>
          <button class="se-prop-toggle">▼</button>
        </div>
        <div class="se-prop-content visible">
          <div class="se-prop-row">
            <span class="se-prop-label">Color</span>
            <div class="se-prop-inputs">
              <input type="color" class="se-prop-input se-mat-color" value="${color}" style="width: 100%; height: 32px; padding: 0;" />
            </div>
          </div>
          <div class="se-prop-row">
            <span class="se-prop-label">Metalness</span>
            <div class="se-prop-inputs">
              <input type="range" class="se-prop-input se-mat-metalness" min="0" max="1" step="0.01" value="${metalness}" style="width: 100%;" />
            </div>
          </div>
          <div class="se-prop-row">
            <span class="se-prop-label">Roughness</span>
            <div class="se-prop-inputs">
              <input type="range" class="se-prop-input se-mat-roughness" min="0" max="1" step="0.01" value="${roughness}" style="width: 100%;" />
            </div>
          </div>
        </div>
      </div>
    `;
  }

  /**
   * Create info section
   */
  createInfoSection(id, obj, data) {
    const childCount = obj.children?.length || 0;

    return `
      <div class="se-prop-section">
        <div class="se-prop-header">
          <div class="se-prop-title"><span>ℹ️</span> Info</div>
          <button class="se-prop-toggle">▼</button>
        </div>
        <div class="se-prop-content visible">
          <div class="se-prop-value"><span class="se-prop-value-label">ID</span><span class="se-prop-value-value">${id}</span></div>
          <div class="se-prop-value"><span class="se-prop-value-label">Type</span><span class="se-prop-value-value">${data.type || obj.type}</span></div>
          <div class="se-prop-value"><span class="se-prop-value-label">Children</span><span class="se-prop-value-value">${childCount}</span></div>
          <div class="se-prop-value"><span class="se-prop-value-label">Visible</span><span class="se-prop-value-value">${obj.visible !== false ? "Yes" : "No"}</span></div>
          ${data.path ? `<div class="se-prop-value"><span class="se-prop-value-label">Path</span><span class="se-prop-value-value" style="font-family: monospace;">${data.path.split("/").pop()}</span></div>` : ""}
        </div>
      </div>
    `;
  }

  /**
   * Create actions section
   */
  createActionsSection(id, obj, data) {
    return `
      <div class="se-prop-section">
        <div class="se-prop-header">
          <div class="se-prop-title"><span>⚡</span> Actions</div>
          <button class="se-prop-toggle">▼</button>
        </div>
        <div class="se-prop-content visible">
          <div class="se-action-buttons">
            <button class="se-action-button primary" id="se-focus-btn">🎯 Focus</button>
            <button class="se-action-button secondary" id="se-copy-btn">📋 Copy</button>
            <button class="se-action-button secondary" id="se-duplicate-btn">🔁 Duplicate</button>
            <button class="se-action-button danger" id="se-delete-btn">🗑️ Delete</button>
          </div>
        </div>
      </div>
    `;
  }

  /**
   * Bind inspector input events
   */
  bindInspectorInputs(id, obj) {
    // Transform inputs
    ["x", "y", "z"].forEach(axis => {
      const posInput = document.querySelector(`.se-pos-${axis}`);
      const rotInput = document.querySelector(`.se-rot-${axis}`);
      const scaleInput = document.querySelector(`.se-scale-${axis}`);

      if (posInput) this.bindTransformInput(id, obj, "position", axis, posInput);
      if (rotInput) this.bindTransformInput(id, obj, "rotation", axis, rotInput);
      if (scaleInput) this.bindTransformInput(id, obj, "scale", axis, scaleInput);
    });

    // Material inputs
    const colorInput = document.querySelector(".se-mat-color");
    const metalnessInput = document.querySelector(".se-mat-metalness");
    const roughnessInput = document.querySelector(".se-mat-roughness");

    if (colorInput) {
      colorInput.addEventListener("input", (e) => {
        const mat = obj.material || obj.children?.[0]?.material;
        if (mat) mat.color.set(e.target.value);
      });
    }

    if (metalnessInput) {
      metalnessInput.addEventListener("input", (e) => {
        const mat = obj.material || obj.children?.[0]?.material;
        if (mat) mat.metalness = parseFloat(e.target.value);
      });
    }

    if (roughnessInput) {
      roughnessInput.addEventListener("input", (e) => {
        const mat = obj.material || obj.children?.[0]?.material;
        if (mat) mat.roughness = parseFloat(e.target.value);
      });
    }

    // Action buttons
    document.getElementById("se-focus-btn")?.addEventListener("click", () => this.focusOnSelected());
    document.getElementById("se-copy-btn")?.addEventListener("click", () => this.copyTransform(obj));
    document.getElementById("se-duplicate-btn")?.addEventListener("click", () => this.duplicateSelected());
    document.getElementById("se-delete-btn")?.addEventListener("click", () => this.deleteSelected());

    // Section toggles
    document.querySelectorAll(".se-prop-header").forEach(header => {
      header.addEventListener("click", () => {
        const content = header.nextElementSibling;
        const toggle = header.querySelector(".se-prop-toggle");
        content.classList.toggle("visible");
        toggle.textContent = content.classList.contains("visible") ? "▼" : "▶";
      });
    });
  }

  /**
   * Bind transform input with undo/redo
   */
  bindTransformInput(id, obj, prop, axis, input) {
    let oldValue = obj[prop][axis];
    let newValue = oldValue;
    let isDragging = false;

    input.addEventListener("mousedown", () => {
      oldValue = obj[prop][axis];
      isDragging = true;
    });

    input.addEventListener("input", (e) => {
      newValue = parseFloat(e.target.value);
      if (!isNaN(newValue)) {
        obj[prop][axis] = newValue;
      }
    });

    input.addEventListener("change", (e) => {
      if (isDragging && oldValue !== newValue) {
        const oldTransform = { [prop]: { [axis]: oldValue } };
        const newTransform = { [prop]: { [axis]: newValue } };
        this.executeCommand(new TransformCommand(this, id, oldTransform, newTransform));
      }
      isDragging = false;
    });
  }

  /**
   * Create a new object
   */
  createObject(type, options = {}) {
    const id = options.id || `object_${this.nextObjectId++}`;
    let object;
    let data = { type, ...options };

    switch (type) {
      case "primitive":
        const primitive = PRIMITIVES[options.shape];
        if (primitive) {
          object = primitive.create();
          object.position.set(
            options.position?.x ?? 0,
            options.position?.y ?? 1,
            options.position?.z ?? 0
          );
          data.type = "primitive";
          data.description = `${primitive.name} ${this.nextObjectId - 1}`;
        }
        break;

      case "light":
        const lightType = LIGHT_TYPES[options.type];
        if (lightType) {
          object = lightType.create();
          object.position.set(
            options.position?.x ?? 0,
            options.position?.y ?? 2,
            options.position?.z ?? 0
          );
          data.type = "light";
          data.lightType = options.type;
          data.description = `${lightType.name} ${this.nextObjectId - 1}`;
        }
        break;

      case "models":
      case "gltf":
        // For now, create placeholder
        object = new THREE.Group();
        object.position.set(
          options.position?.x ?? 0,
          options.position?.y ?? 0,
          options.position?.z ?? 0
        );
        data.path = options.path;
        data.description = options.name || id;
        break;

      case "splats":
        object = new THREE.Group();
        object.position.set(0.35, 1.0, 1.9);
        object.rotation.set(0, Math.PI, Math.PI);
        data.path = options.path;
        data.description = options.name || id;
        break;

      default:
        object = new THREE.Group();
        object.position.set(0, 0, 0);
    }

    if (!object) {
      this.logger.warn(`Failed to create object: ${type}`);
      return null;
    }

    // Add to scene
    this.sceneManager?.scene?.add(object);

    // Register with gizmo
    if (this.gizmoManager) {
      this.gizmoManager.registerObject(object, id, type);
    }

    // Store in scene objects
    this.sceneObjects.set(id, { id, object, data, type, source: "editor" });

    // Refresh UI
    this.refreshSceneGraph();
    this.selectObject(id);

    this.logger.log(`Created ${type}: ${id}`);
    this.setStatus(`Created ${data.description || id}`);

    return { id, object, data };
  }

  /**
   * Delete selected object
   */
  deleteSelected() {
    if (!this.selectedObjectId) return;

    this.executeCommand(new DeleteObjectCommand(this, this.selectedObjectId));
  }

  /**
   * Delete an object
   */
  deleteObject(id) {
    const objData = this.sceneObjects.get(id);
    if (!objData) return;

    const { object, source } = objData;

    // Remove from scene
    if (object.parent) {
      object.parent.remove(object);
    }

    // Unregister gizmo
    if (this.gizmoManager) {
      this.gizmoManager.unregisterObject(object);
    }

    // Remove from sceneManager if needed
    if (source === "scene" && this.sceneManager) {
      this.sceneManager.removeObject(id);
    }

    // Remove from tracking
    this.sceneObjects.delete(id);

    // Deselect if current
    if (this.selectedObjectId === id) {
      this.selectedObjectId = null;
    }

    this.refreshSceneGraph();
    this.updateInspector(null);
    this.setStatus(`Deleted ${id}`);
  }

  /**
   * Duplicate selected object
   */
  duplicateSelected() {
    if (!this.selectedObjectId) return;

    const objData = this.sceneObjects.get(this.selectedObjectId);
    if (!objData) return;

    // Create clone
    const clone = objData.object.clone();
    clone.position.x += 1;

    const newId = `${this.selectedObjectId}_copy`;
    clone.name = newId;

    // Add to scene
    this.sceneManager?.scene?.add(clone);

    // Register gizmo
    if (this.gizmoManager) {
      this.gizmoManager.registerObject(clone, newId, objData.type);
    }

    // Store
    const newData = { ...objData.data, description: `${objData.data.description || objData.id} (copy)` };
    this.sceneObjects.set(newId, { id: newId, object: clone, data: newData, type: objData.type, source: "editor" });

    this.refreshSceneGraph();
    this.selectObject(newId);
    this.setStatus(`Duplicated as ${newId}`);
  }

  /**
   * Get object by ID
   */
  getObjectById(id) {
    return this.sceneObjects.get(id) || null;
  }

  /**
   * Get object icon
   */
  getObjectIcon(type) {
    const icons = {
      splat: "🌐",
      gltf: "📦",
      light: "💡",
      video: "🎬",
      collider: "🔷",
      primitive: "🔷",
      spawned: "✨"
    };
    return icons[type] || "📄";
  }

  /**
   * Execute a command (with undo/redo)
   */
  executeCommand(command) {
    command.execute();
    this.undoStack.push(command);
    this.redoStack = []; // Clear redo stack
    this.updateUndoRedoButtons();
  }

  /**
   * Undo
   */
  undo() {
    if (this.undoStack.length === 0) return;

    const command = this.undoStack.pop();
    command.undo();
    this.redoStack.push(command);
    this.updateUndoRedoButtons();
    this.setStatus("Undo");
    this.refreshSceneGraph();
  }

  /**
   * Redo
   */
  redo() {
    if (this.redoStack.length === 0) return;

    const command = this.redoStack.pop();
    command.execute();
    this.undoStack.push(command);
    this.updateUndoRedoButtons();
    this.setStatus("Redo");
    this.refreshSceneGraph();
  }

  /**
   * Update undo/redo button states
   */
  updateUndoRedoButtons() {
    const undoBtn = document.getElementById("se-undo-btn");
    const redoBtn = document.getElementById("se-redo-btn");

    if (undoBtn) undoBtn.disabled = this.undoStack.length === 0;
    if (redoBtn) redoBtn.disabled = this.redoStack.length === 0;
  }

  /**
   * Focus camera on selected object
   */
  focusOnSelected() {
    if (!this.selectedObjectId || !this.characterController) return;

    const obj = this.getObjectById(this.selectedObjectId);
    if (!obj) return;

    const targetPos = obj.object.position.clone();

    if (this.characterController.character) {
      this.characterController.character.setTranslation(
        { x: targetPos.x, y: Math.max(0.9, targetPos.y), z: targetPos.z + 5 },
        true
      );
    }

    this.setStatus(`Focused on ${this.selectedObjectId}`);
  }

  /**
   * Copy transform to clipboard
   */
  copyTransform(obj) {
    const transform = {
      position: {
        x: obj.object.position.x.toFixed(2),
        y: obj.object.position.y.toFixed(2),
        z: obj.object.position.z.toFixed(2)
      },
      rotation: {
        x: obj.object.rotation.x.toFixed(4),
        y: obj.object.rotation.y.toFixed(4),
        z: obj.object.rotation.z.toFixed(4)
      },
      scale: {
        x: obj.object.scale.x.toFixed(2),
        y: obj.object.scale.y.toFixed(2),
        z: obj.object.scale.z.toFixed(2)
      }
    };

    const text = JSON.stringify(transform, null, 2);
    navigator.clipboard.writeText(text).then(() => {
      this.setStatus("Copied transform to clipboard");
    });

    console.log("Transform:", transform);
  }

  /**
   * Toggle play mode
   */
  togglePlayMode() {
    this.isPlaying = !this.isPlaying;

    const btn = document.getElementById("se-play-btn");
    const icon = btn.querySelector(".se-play-icon");
    const text = btn.querySelector(".se-play-text");

    if (this.isPlaying) {
      btn.classList.add("playing");
      icon.textContent = "⏸";
      text.textContent = "PAUSE";
      this.setStatus("Play mode active");
    } else {
      btn.classList.remove("playing");
      icon.textContent = "▶";
      text.textContent = "PLAY";
      this.setStatus("Edit mode active");
    }
  }

  /**
   * Set tool
   */
  setTool(tool) {
    document.querySelectorAll(".se-tool-btn").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.tool === tool);
    });

    if (this.gizmoManager) {
      switch (tool) {
        case "select":
          // Just select
          break;
        case "move":
          this.gizmoManager.currentMode = "translate";
          this.gizmoManager.controls.forEach(c => c.setMode("translate"));
          break;
        case "rotate":
          this.gizmoManager.currentMode = "rotate";
          this.gizmoManager.controls.forEach(c => c.setMode("rotate"));
          break;
        case "scale":
          this.gizmoManager.currentMode = "scale";
          this.gizmoManager.controls.forEach(c => c.setMode("scale"));
          break;
      }
    }

    this.setStatus(`Tool: ${tool}`);
  }

  /**
   * Update stats
   */
  updateStats() {
    const stats = document.getElementById("se-stats");
    if (stats) {
      stats.textContent = `${this.sceneObjects.size} objects`;
    }
  }

  /**
   * Set status message
   */
  setStatus(message) {
    const status = document.getElementById("se-status");
    if (status) {
      status.textContent = message;
      setTimeout(() => {
        if (status.textContent === message) {
          status.textContent = this.isPlaying ? "Playing..." : "Ready";
        }
      }, 3000);
    }
  }

  /**
   * Save object data (for undo)
   */
  saveObjectData(id) {
    const obj = this.sceneObjects.get(id);
    if (!obj) return null;

    return {
      id: obj.id,
      data: { ...obj.data },
      transform: {
        position: { x: obj.object.position.x, y: obj.object.position.y, z: obj.object.position.z },
        rotation: { x: obj.object.rotation.x, y: obj.object.rotation.y, z: obj.object.rotation.z },
        scale: { x: obj.object.scale.x, y: obj.object.scale.y, z: obj.object.scale.z }
      },
      visible: obj.object.visible
    };
  }

  /**
   * Restore object data (for undo)
   */
  restoreObjectData(savedData) {
    // Recreate object
    const { id, data, transform, visible } = savedData;

    let object;

    if (data.type === "primitive" && PRIMITIVES[data.shape]) {
      object = PRIMITIVES[data.shape].create();
    } else if (data.type === "light" && LIGHT_TYPES[data.lightType]) {
      object = LIGHT_TYPES[data.lightType].create();
    } else {
      object = new THREE.Group();
    }

    object.position.set(transform.position.x, transform.position.y, transform.position.z);
    object.rotation.set(transform.rotation.x, transform.rotation.y, transform.rotation.z);
    object.scale.set(transform.scale.x, transform.scale.y, transform.scale.z);
    object.visible = visible;

    this.sceneManager?.scene?.add(object);

    if (this.gizmoManager) {
      this.gizmoManager.registerObject(object, id, data.type);
    }

    this.sceneObjects.set(id, { id, object, data, type: data.type, source: "editor" });

    this.refreshSceneGraph();
  }

  /**
   * Show save dialog
   */
  showSaveDialog() {
    const sceneData = this.serializeScene();
    const json = JSON.stringify(sceneData, null, 2);

    // Create download
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `scene_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);

    this.setStatus("Scene saved to file");
  }

  /**
   * Show load dialog
   */
  showLoadDialog() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";

    input.onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const sceneData = JSON.parse(event.target.result);
          this.loadScene(sceneData);
        } catch (err) {
          this.logger.error("Failed to load scene:", err);
        }
      };
      reader.readAsText(file);
    };

    input.click();
  }

  /**
   * Serialize scene to JSON
   */
  serializeScene() {
    const objects = [];

    this.sceneObjects.forEach((obj) => {
      objects.push({
        id: obj.id,
        type: obj.data.type,
        description: obj.data.description || obj.id,
        path: obj.data.path,
        position: {
          x: obj.object.position.x,
          y: obj.object.position.y,
          z: obj.object.position.z
        },
        rotation: {
          x: obj.object.rotation.x,
          y: obj.object.rotation.y,
          z: obj.object.rotation.z
        },
        scale: {
          x: obj.object.scale.x,
          y: obj.object.scale.y,
          z: obj.object.scale.z
        },
        visible: obj.object.visible
      });
    });

    return {
      version: "1.0",
      name: "Shadow Scene",
      timestamp: new Date().toISOString(),
      objects
    };
  }

  /**
   * Load scene from JSON
   */
  loadScene(sceneData) {
    // Clear existing objects
    this.sceneObjects.forEach((obj) => {
      if (obj.source === "editor" && obj.object.parent) {
        obj.object.parent.remove(obj.object);
      }
    });
    this.sceneObjects.clear();

    // Load objects
    sceneData.objects.forEach((objData) => {
      let object;

      if (objData.type === "light" && LIGHT_TYPES[objData.lightType]) {
        object = LIGHT_TYPES[objData.lightType].create();
      } else {
        object = new THREE.Group();
      }

      object.position.set(objData.position.x, objData.position.y, objData.position.z);
      object.rotation.set(objData.rotation.x, objData.rotation.y, objData.rotation.z);
      object.scale.set(objData.scale.x, objData.scale.y, objData.scale.z);
      object.visible = objData.visible;

      this.sceneManager?.scene?.add(object);

      if (this.gizmoManager) {
        this.gizmoManager.registerObject(object, objData.id, objData.type);
      }

      this.sceneObjects.set(objData.id, {
        id: objData.id,
        object,
        data: objData,
        type: objData.type,
        source: "editor"
      });
    });

    this.refreshSceneGraph();
    this.setStatus(`Loaded ${sceneData.objects.length} objects`);
  }

  /**
   * Export to sceneData.js format
   */
  exportToCode() {
    const lines = [];
    lines.push("// Auto-generated scene data from Shadow Editor");
    lines.push("// Generated: " + new Date().toISOString());
    lines.push("");
    lines.push("export const sceneObjects = {");

    this.sceneObjects.forEach((obj) => {
      const { id, data, object } = obj;

      lines.push(`  ${id}: {`);
      lines.push(`    id: "${id}",`);
      lines.push(`    type: "${data.type || "gltf"}",`);
      if (data.path) lines.push(`    path: "${data.path}",`);
      lines.push(`    description: "${data.description || id}",`);
      lines.push(`    position: { x: ${object.position.x.toFixed(2)}, y: ${object.position.y.toFixed(2)}, z: ${object.position.z.toFixed(2)} },`);
      lines.push(`    rotation: { x: ${object.rotation.x.toFixed(4)}, y: ${object.rotation.y.toFixed(4)}, z: ${object.rotation.z.toFixed(4)} },`);
      lines.push(`    scale: { x: ${object.scale.x.toFixed(2)}, y: ${object.scale.y.toFixed(2)}, z: ${object.scale.z.toFixed(2)} }`);
      lines.push(`  },`);
    });

    lines.push("};");

    const code = lines.join("\n");

    // Download
    const blob = new Blob([code], { type: "text/javascript" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "sceneData.js";
    a.click();
    URL.revokeObjectURL(url);

    // Also log to console
    console.log("=== Generated sceneData.js ===");
    console.log(code);

    this.setStatus("Exported to sceneData.js");
  }

  /**
   * Open editor
   */
  open() {
    if (this.isOpen) return;

    this.isOpen = true;
    this.element.classList.remove("hidden");

    if (document.pointerLockElement) {
      document.exitPointerLock();
    }

    this.refreshSceneData();
    this.setStatus("Editor opened");
  }

  /**
   * Close editor
   */
  close() {
    if (!this.isOpen) return;

    this.isOpen = false;
    this.element.classList.add("hidden");
    this.setStatus("Editor closed");
  }

  /**
   * Toggle open/close
   */
  toggle() {
    if (this.isOpen) {
      this.close();
    } else {
      this.open();
    }
  }

  /**
   * Clean up
   */
  destroy() {
    if (this.keydownHandler) {
      window.removeEventListener("keydown", this.keydownHandler);
    }

    if (this.element && this.element.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }

    const styles = document.getElementById("shadow-editor-styles");
    if (styles && styles.parentNode) {
      styles.parentNode.removeChild(styles);
    }

    if (window.shadowEditor === this) {
      delete window.shadowEditor;
    }
  }
}

export default ShadowEditor;
