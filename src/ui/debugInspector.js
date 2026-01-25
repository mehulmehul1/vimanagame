/**
 * debugInspector.js - BLENDER-STYLE DEBUG INSPECTOR
 * =============================================================================
 *
 * ROLE: Provides a Blender-inspired debug interface for inspecting and
 * modifying scene objects during development. Features hierarchical outliner,
 * properties panel, layer visibility controls, and gizmo integration.
 *
 * KEY RESPONSIBILITIES:
 * - Display scene hierarchy in outliner (like Blender's Outliner)
 * - Show/edit object properties (Transform, Material, Visibility)
 * - Layer visibility toggles for scene organization
 * - Object selection (click in outliner or 3D view)
 * - Integration with GizmoManager for visual manipulation
 * - Search and filter functionality
 *
 * KEYBOARD CONTROLS:
 * - Ctrl+Shift+D: Toggle Debug Inspector
 * - Delete: Remove selected object
 * - F: Focus camera on selected object
 * - G/R/S: Translate/Rotate/Scale (via GizmoManager)
 *
 * USAGE:
 *   const debugInspector = new DebugInspector({
 *     sceneManager,
 *     gizmoManager,
 *     camera,
 *     renderer
 *   });
 *
 * =============================================================================
 */

import { Logger } from "../utils/logger.js";

// Layer categories for organizing scene objects (like Blender collections)
const LAYER_CATEGORIES = {
  splats: { name: "Splat Scenes", icon: "🌐", color: "#e74c3c" },
  gltf: { name: "GLTF Models", icon: "📦", color: "#3498db" },
  lights: { name: "Lights", icon: "💡", color: "#f1c40f" },
  videos: { name: "Videos", icon: "🎬", color: "#9b59b6" },
  colliders: { name: "Colliders", icon: "🔷", color: "#1abc9c" },
  effects: { name: "VFX", icon: "✨", color: "#e91e63" },
  other: { name: "Other", icon: "📄", color: "#95a5a6" }
};

class DebugInspector {
  constructor(options = {}) {
    this.sceneManager = options.sceneManager || null;
    this.gizmoManager = options.gizmoManager || null;
    this.camera = options.camera || null;
    this.renderer = options.renderer || null;
    this.characterController = options.characterController || null;

    this.logger = new Logger("DebugInspector", false);

    // UI state
    this.isOpen = false;
    this.selectedObjectId = null;
    this.expandedObjects = new Set(); // Track which objects are expanded in outliner
    this.layerVisibility = {}; // Track layer visibility
    this.searchQuery = "";
    this.filterType = "all"; // all, splats, gltf, lights, videos

    // Event handlers (stored for cleanup)
    this.eventHandlers = {};

    // Build UI
    this.createInspectorHTML();
    this.injectStyles();
    this.bindEvents();

    // Initialize layer visibility (all visible by default)
    Object.keys(LAYER_CATEGORIES).forEach(layer => {
      this.layerVisibility[layer] = true;
    });

    // Register global keyboard shortcut
    this.bindKeyboardShortcuts();

    // Make self available globally for debugging
    window.debugInspector = this;

    this.logger.log("DebugInspector initialized");
  }

  /**
   * Create the HTML structure for the debug inspector
   */
  createInspectorHTML() {
    const inspector = document.createElement("div");
    inspector.id = "debug-inspector";
    inspector.className = "debug-inspector hidden";

    inspector.innerHTML = `
      <div class="di-overlay"></div>

      <div class="di-container">
        <!-- Header -->
        <div class="di-header">
          <div class="di-title">
            <span class="di-icon">🔧</span>
            <span>Debug Inspector</span>
            <span class="di-object-count" id="di-object-count">0 objects</span>
          </div>
          <div class="di-controls">
            <button class="di-btn" id="di-refresh" title="Refresh (R)">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M23 4v6h-6M1 20v-6h6"/>
                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
              </svg>
            </button>
            <button class="di-btn" id="di-compact" title="Compact Mode">□</button>
            <button class="di-btn di-close" id="di-close" title="Close (Ctrl+Shift+D)">×</button>
          </div>
        </div>

        <!-- Search Bar -->
        <div class="di-search">
          <input type="text" id="di-search-input" placeholder="Search objects..." />
          <select id="di-filter-type">
            <option value="all">All Types</option>
            <option value="splat">Splats</option>
            <option value="gltf">Models</option>
            <option value="light">Lights</option>
            <option value="video">Videos</option>
            <option value="collider">Colliders</option>
          </select>
        </div>

        <!-- Main Content Area -->
        <div class="di-content">
          <!-- Outliner Panel -->
          <div class="di-panel di-outliner-panel">
            <div class="di-panel-header">
              <h3>Outliner</h3>
              <div class="di-layer-toggles" id="di-layer-toggles"></div>
            </div>
            <div class="di-outliner" id="di-outliner">
              <div class="di-empty">No objects loaded</div>
            </div>
          </div>

          <!-- Properties Panel -->
          <div class="di-panel di-properties-panel">
            <div class="di-panel-header">
              <h3>Properties</h3>
            </div>
            <div class="di-properties" id="di-properties">
              <div class="di-empty">Select an object to view properties</div>
            </div>
          </div>
        </div>

        <!-- Status Bar -->
        <div class="di-status-bar" id="di-status-bar">
          <span id="di-status">Ready</span>
          <span class="di-shortcuts">Ctrl+Shift+D: Toggle | G/R/S: Transform | Del: Remove</span>
        </div>
      </div>
    `;

    document.body.appendChild(inspector);
    this.element = inspector;
  }

  /**
   * Inject CSS styles for the debug inspector
   */
  injectStyles() {
    const styleId = "debug-inspector-styles";
    if (document.getElementById(styleId)) return;

    const style = document.createElement("style");
    style.id = styleId;
    style.textContent = `
      /* ===== Debug Inspector Base Styles ===== */
      .debug-inspector {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 10000;
        pointer-events: none;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 13px;
        color: #e0e0e0;
      }

      .debug-inspector.hidden {
        display: none;
      }

      .debug-inspector:not(.hidden) {
        display: block;
      }

      .di-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        pointer-events: auto;
      }

      /* Container */
      .di-container {
        position: absolute;
        top: 20px;
        left: 20px;
        bottom: 20px;
        width: 700px;
        max-width: calc(100vw - 40px);
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.1);
        display: flex;
        flex-direction: column;
        overflow: hidden;
        pointer-events: auto;
      }

      .di-container.compact {
        width: 350px;
      }

      .di-container.compact .di-properties-panel {
        display: none;
      }

      /* Header */
      .di-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 16px;
        background: rgba(0, 0, 0, 0.3);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      }

      .di-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        font-size: 14px;
      }

      .di-icon {
        font-size: 18px;
      }

      .di-object-count {
        font-size: 11px;
        color: #888;
        font-weight: normal;
        margin-left: 8px;
      }

      .di-controls {
        display: flex;
        gap: 6px;
      }

      .di-btn {
        width: 28px;
        height: 28px;
        border: none;
        background: rgba(255, 255, 255, 0.1);
        color: #ccc;
        border-radius: 6px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
      }

      .di-btn:hover {
        background: rgba(255, 255, 255, 0.2);
        color: #fff;
      }

      .di-btn.di-close:hover {
        background: #e74c3c;
      }

      /* Search Bar */
      .di-search {
        display: flex;
        gap: 8px;
        padding: 12px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      }

      .di-search input {
        flex: 1;
        padding: 8px 12px;
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        color: #e0e0e0;
        font-size: 12px;
      }

      .di-search input:focus {
        outline: none;
        border-color: #3498db;
      }

      .di-search select {
        padding: 8px 10px;
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        color: #e0e0e0;
        font-size: 12px;
        cursor: pointer;
      }

      /* Content Area */
      .di-content {
        display: flex;
        flex: 1;
        overflow: hidden;
      }

      .di-panel {
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }

      .di-outliner-panel {
        flex: 1;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        min-width: 0;
      }

      .di-properties-panel {
        width: 320px;
        flex-shrink: 0;
      }

      .di-container.compact .di-outliner-panel {
        border-right: none;
      }

      /* Panel Headers */
      .di-panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 16px;
        background: rgba(0, 0, 0, 0.2);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      }

      .di-panel-header h3 {
        margin: 0;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888;
      }

      /* Layer Toggles */
      .di-layer-toggles {
        display: flex;
        gap: 4px;
      }

      .di-layer-toggle {
        width: 22px;
        height: 22px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        transition: all 0.2s;
        opacity: 0.6;
      }

      .di-layer-toggle:hover {
        opacity: 1;
      }

      .di-layer-toggle.active {
        opacity: 1;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.3);
      }

      .di-layer-toggle.hidden {
        opacity: 0.3;
        text-decoration: line-through;
      }

      /* Outliner */
      .di-outliner {
        flex: 1;
        overflow-y: auto;
        padding: 8px 0;
      }

      .di-outliner::-webkit-scrollbar {
        width: 8px;
      }

      .di-outliner::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
      }

      .di-outliner::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 4px;
      }

      .di-empty {
        padding: 40px 20px;
        text-align: center;
        color: #666;
        font-style: italic;
      }

      /* Object Items */
      .di-layer {
        margin-bottom: 8px;
      }

      .di-layer-header {
        display: flex;
        align-items: center;
        padding: 6px 16px;
        cursor: pointer;
        user-select: none;
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid transparent;
      }

      .di-layer-header:hover {
        background: rgba(255, 255, 255, 0.06);
      }

      .di-layer-header .di-layer-icon {
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 8px;
        border-radius: 4px;
      }

      .di-layer-header .di-layer-name {
        flex: 1;
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      .di-layer-header .di-layer-count {
        font-size: 10px;
        color: #666;
        background: rgba(0, 0, 0, 0.3);
        padding: 2px 6px;
        border-radius: 10px;
      }

      .di-layer-toggle-btn {
        width: 20px;
        height: 20px;
        border: none;
        background: none;
        color: #888;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 4px;
        padding: 0;
      }

      .di-layer-toggle-btn:hover {
        color: #fff;
      }

      .di-layer-objects {
        display: none;
      }

      .di-layer-objects.visible {
        display: block;
      }

      /* Object Item */
      .di-object-item {
        display: flex;
        align-items: center;
        padding: 6px 16px 6px 44px;
        cursor: pointer;
        user-select: none;
        transition: all 0.15s;
        border-left: 3px solid transparent;
      }

      .di-object-item:hover {
        background: rgba(255, 255, 255, 0.05);
      }

      .di-object-item.selected {
        background: rgba(52, 152, 219, 0.2);
        border-left-color: #3498db;
      }

      .di-object-item .di-expand-btn {
        width: 16px;
        height: 16px;
        border: none;
        background: none;
        color: #666;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 4px;
        padding: 0;
        font-size: 10px;
      }

      .di-object-item .di-expand-btn.expanded {
        transform: rotate(90deg);
      }

      .di-object-item .di-object-icon {
        width: 16px;
        height: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 8px;
        font-size: 12px;
      }

      .di-object-item .di-object-name {
        flex: 1;
        font-size: 12px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .di-object-item .di-object-type {
        font-size: 9px;
        color: #666;
        text-transform: uppercase;
        padding: 2px 4px;
        border-radius: 3px;
        background: rgba(0, 0, 0, 0.3);
      }

      .di-object-item .di-object-visible {
        width: 20px;
        height: 20px;
        border: none;
        background: none;
        color: #666;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-left: 4px;
      }

      .di-object-item .di-object-visible:hover {
        color: #fff;
      }

      .di-object-item .di-object-visible.hidden {
        color: #e74c3c;
      }

      /* Object Children */
      .di-object-children {
        display: none;
        background: rgba(0, 0, 0, 0.2);
      }

      .di-object-children.visible {
        display: block;
      }

      .di-object-child {
        display: flex;
        align-items: center;
        padding: 4px 16px 4px 60px;
        cursor: pointer;
        font-size: 11px;
        color: #888;
      }

      .di-object-child:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #ccc;
      }

      .di-object-child .di-child-icon {
        margin-right: 6px;
      }

      /* Properties Panel */
      .di-properties {
        flex: 1;
        overflow-y: auto;
        padding: 12px;
      }

      .di-properties::-webkit-scrollbar {
        width: 6px;
      }

      .di-properties::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
      }

      .di-properties::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 3px;
      }

      .di-prop-section {
        margin-bottom: 16px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 8px;
        overflow: hidden;
      }

      .di-prop-section-header {
        display: flex;
        align-items: center;
        padding: 10px 12px;
        background: rgba(255, 255, 255, 0.05);
        cursor: pointer;
        user-select: none;
      }

      .di-prop-section-header:hover {
        background: rgba(255, 255, 255, 0.08);
      }

      .di-prop-section-header .di-prop-icon {
        margin-right: 8px;
      }

      .di-prop-section-header .di-prop-title {
        flex: 1;
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      .di-prop-section-header .di-prop-toggle {
        width: 20px;
        height: 20px;
        border: none;
        background: none;
        color: #888;
        cursor: pointer;
      }

      .di-prop-section-content {
        display: none;
        padding: 12px;
      }

      .di-prop-section-content.visible {
        display: block;
      }

      /* Property Rows */
      .di-prop-row {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
      }

      .di-prop-row:last-child {
        margin-bottom: 0;
      }

      .di-prop-label {
        width: 70px;
        font-size: 11px;
        color: #888;
      }

      .di-prop-inputs {
        flex: 1;
        display: flex;
        gap: 6px;
      }

      .di-prop-input-group {
        flex: 1;
        display: flex;
        align-items: center;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 4px;
        overflow: hidden;
      }

      .di-prop-input-group .di-prop-axis {
        width: 20px;
        font-size: 10px;
        text-align: center;
        color: #666;
        background: rgba(0, 0, 0, 0.5);
      }

      .di-prop-input-group.x .di-prop-axis { color: #e74c3c; }
      .di-prop-input-group.y .di-prop-axis { color: #2ecc71; }
      .di-prop-input-group.z .di-prop-axis { color: #3498db; }
      .di-prop-input-group.w .di-prop-axis { color: #f39c12; }

      .di-prop-input {
        width: 100%;
        padding: 6px 8px;
        background: transparent;
        border: none;
        color: #e0e0e0;
        font-size: 11px;
        font-family: 'Consolas', 'Monaco', monospace;
      }

      .di-prop-input:focus {
        outline: none;
        background: rgba(52, 152, 219, 0.2);
      }

      /* Action Buttons */
      .di-prop-actions {
        display: flex;
        gap: 8px;
        margin-top: 12px;
      }

      .di-action-btn {
        flex: 1;
        padding: 8px 12px;
        border: none;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
      }

      .di-action-btn:focus {
        outline: none;
      }

      .di-action-btn.primary {
        background: #3498db;
        color: #fff;
      }

      .di-action-btn.primary:hover {
        background: #2980b9;
      }

      .di-action-btn.danger {
        background: #e74c3c;
        color: #fff;
      }

      .di-action-btn.danger:hover {
        background: #c0392b;
      }

      .di-action-btn.secondary {
        background: rgba(255, 255, 255, 0.1);
        color: #ccc;
      }

      .di-action-btn.secondary:hover {
        background: rgba(255, 255, 255, 0.2);
        color: #fff;
      }

      /* Status Bar */
      .di-status-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 16px;
        background: rgba(0, 0, 0, 0.3);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 10px;
        color: #666;
      }

      .di-shortcuts {
        color: #888;
      }

      /* Value display badge */
      .di-value-badge {
        display: inline-block;
        padding: 2px 6px;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 4px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 10px;
        color: #888;
      }

      .di-prop-value {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      }

      .di-prop-value:last-child {
        border-bottom: none;
      }

      .di-prop-value-label {
        font-size: 11px;
        color: #888;
      }

      .di-prop-value-value {
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 11px;
        color: #e0e0e0;
      }

      /* Highlight matching search */
      .di-object-item.match .di-object-name,
      .di-object-child.match {
        background: rgba(241, 196, 15, 0.3);
        border-radius: 3px;
        padding: 2px 4px;
        margin: -2px -4px;
      }

      /* Checkbox */
      .di-checkbox {
        width: 16px;
        height: 16px;
        cursor: pointer;
      }

      /* Responsive */
      @media (max-width: 768px) {
        .di-container {
          width: calc(100vw - 20px);
          left: 10px;
          right: 10px;
          top: 10px;
          bottom: 10px;
        }

        .di-properties-panel {
          display: none;
        }

        .di-container:not(.compact) .di-properties-panel {
          display: flex;
          position: absolute;
          top: 120px;
          left: 10px;
          right: 10px;
          bottom: 50px;
          z-index: 10;
          background: #1a1a2e;
        }
      }
    `;

    document.head.appendChild(style);
  }

  /**
   * Bind event listeners
   */
  bindEvents() {
    // Close button
    const closeBtn = document.getElementById("di-close");
    closeBtn?.addEventListener("click", () => this.close());

    // Overlay click to close
    const overlay = this.element.querySelector(".di-overlay");
    overlay?.addEventListener("click", () => this.close());

    // Refresh button
    const refreshBtn = document.getElementById("di-refresh");
    refreshBtn?.addEventListener("click", () => {
      this.refreshOutliner();
      this.setStatus("Refreshed");
    });

    // Compact mode toggle
    const compactBtn = document.getElementById("di-compact");
    compactBtn?.addEventListener("click", () => {
      this.element.querySelector(".di-container").classList.toggle("compact");
      compactBtn.textContent = compactBtn.textContent === "□" ? "▦" : "□";
    });

    // Search input
    const searchInput = document.getElementById("di-search-input");
    searchInput?.addEventListener("input", (e) => {
      this.searchQuery = e.target.value.toLowerCase();
      this.refreshOutliner();
    });

    // Filter type select
    const filterSelect = document.getElementById("di-filter-type");
    filterSelect?.addEventListener("change", (e) => {
      this.filterType = e.target.value;
      this.refreshOutliner();
    });

    // Object selection via raycasting
    if (this.renderer && this.camera) {
      this.renderer.domElement.addEventListener("dblclick", (e) => this.handleDoubleClick(e));
    }
  }

  /**
   * Bind keyboard shortcuts
   */
  bindKeyboardShortcuts() {
    this.keydownHandler = (e) => {
      // Ignore if typing in input
      if (document.activeElement.tagName === "INPUT" ||
          document.activeElement.tagName === "TEXTAREA") {
        return;
      }

      // Ctrl+Shift+D to toggle
      if (e.ctrlKey && e.shiftKey && (e.key === "D" || e.key === "d")) {
        e.preventDefault();
        this.toggle();
        return;
      }

      // Only handle other keys if inspector is open
      if (!this.isOpen) return;

      // Delete to remove selected object
      if (e.key === "Delete") {
        this.deleteSelectedObject();
        return;
      }

      // F to focus on selected object
      if (e.key === "f" || e.key === "F") {
        this.focusOnSelected();
        return;
      }

      // Escape to deselect
      if (e.key === "Escape") {
        this.deselectObject();
        return;
      }

      // G/R/S for transform (delegated to GizmoManager)
      if (this.gizmoManager && this.selectedObjectId) {
        const obj = this.getObjectById(this.selectedObjectId);
        if (obj) {
          if (e.key === "g" || e.key === "G") {
            this.gizmoManager.currentMode = "translate";
            this.gizmoManager.controls.forEach(c => c.setMode("translate"));
            this.setStatus("Mode: Translate");
          } else if (e.key === "r" || e.key === "R") {
            this.gizmoManager.currentMode = "rotate";
            this.gizmoManager.controls.forEach(c => c.setMode("rotate"));
            this.setStatus("Mode: Rotate");
          } else if (e.key === "s" || e.key === "S") {
            this.gizmoManager.currentMode = "scale";
            this.gizmoManager.controls.forEach(c => c.setMode("scale"));
            this.setStatus("Mode: Scale");
          }
        }
      }
    };

    window.addEventListener("keydown", this.keydownHandler);
  }

  /**
   * Handle double click on canvas to select object
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

    // Get all scene objects
    const objects = [];
    this.sceneManager.objects.forEach((obj) => {
      if (obj.isObject3D) {
        objects.push(obj);
        // Also add children
        obj.traverse((child) => {
          if (child.isObject3D) objects.push(child);
        });
      }
    });

    const intersects = raycaster.intersectObjects(objects, false);

    if (intersects.length > 0) {
      // Find the root object
      let obj = intersects[0].object;
      while (obj.parent && obj.parent.type !== "Scene") {
        obj = obj.parent;
      }

      // Find the object ID
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
   * Get an object by ID from all managers
   */
  getObjectById(id) {
    if (!this.sceneManager) return null;

    // Check sceneManager
    const obj = this.sceneManager.getObject(id);
    if (obj) return { id, object: obj, source: "scene" };

    // Check lightManager
    if (window.lightManager) {
      const light = window.lightManager.getLight(id);
      if (light) return { id, object: light, source: "light" };
    }

    // Check videoManager
    if (window.videoManager) {
      // Video objects are nested
      const player = window.videoManager.videoPlayers.get(id);
      if (player?.videoMesh) {
        return { id, object: player.videoMesh, source: "video" };
      }
    }

    return null;
  }

  /**
   * Refresh the outliner view
   */
  refreshOutliner() {
    const outliner = document.getElementById("di-outliner");
    const layerToggles = document.getElementById("di-layer-toggles");
    if (!outliner || !layerToggles) return;

    // Clear current content
    outliner.innerHTML = "";
    layerToggles.innerHTML = "";

    // Build layer toggles
    Object.entries(LAYER_CATEGORIES).forEach(([key, layer]) => {
      const btn = document.createElement("button");
      btn.className = `di-layer-toggle ${this.layerVisibility[key] ? "active" : "hidden"}`;
      btn.title = `${layer.name} (${key})`;
      btn.innerHTML = layer.icon;
      btn.style.backgroundColor = layer.color;
      btn.addEventListener("click", () => this.toggleLayer(key));
      layerToggles.appendChild(btn);
    });

    // Collect objects by layer
    const layers = {};
    let totalObjects = 0;

    Object.keys(LAYER_CATEGORIES).forEach(key => {
      layers[key] = [];
    });

    // Collect from sceneManager
    if (this.sceneManager) {
      this.sceneManager.objects.forEach((obj, id) => {
        const data = this.sceneManager.objectData.get(id);
        const type = data?.type || "other";

        let layerKey = "other";
        if (type === "splat") layerKey = "splats";
        else if (type === "gltf") layerKey = "gltf";

        // Apply filter
        if (this.filterType !== "all" && type !== this.filterType) return;

        // Apply search
        const name = data?.description || id;
        if (this.searchQuery && !name.toLowerCase().includes(this.searchQuery)) return;

        layers[layerKey].push({
          id,
          object: obj,
          data,
          name,
          visible: obj.visible !== false
        });
        totalObjects++;
      });
    }

    // Collect from lightManager
    if (window.lightManager?.lights) {
      window.lightManager.lights.forEach((light, id) => {
        if (this.filterType !== "all" && this.filterType !== "light") return;
        if (this.searchQuery && !id.toLowerCase().includes(this.searchQuery)) return;

        layers.lights.push({
          id,
          object: light,
          data: { type: "light" },
          name: id,
          visible: light.visible !== false
        });
        totalObjects++;
      });
    }

    // Collect from videoManager
    if (window.videoManager?.videoPlayers) {
      window.videoManager.videoPlayers.forEach((player, id) => {
        if (this.filterType !== "all" && this.filterType !== "video") return;
        if (this.searchQuery && !id.toLowerCase().includes(this.searchQuery)) return;

        layers.videos.push({
          id,
          object: player.videoMesh,
          data: { type: "video" },
          name: id,
          visible: player.visible !== false
        });
        totalObjects++;
      });
    }

    // Collect gizmo objects
    if (this.gizmoManager?.objects) {
      this.gizmoManager.objects.forEach((item) => {
        const alreadyListed =
          layers.splats.some(o => o.id === item.id) ||
          layers.gltf.some(o => o.id === item.id) ||
          layers.lights.some(o => o.id === item.id) ||
          layers.videos.some(o => o.id === item.id);

        if (!alreadyListed) {
          const layerKey = item.type === "spawned" ? "effects" : "colliders";
          if (this.filterType !== "all" && this.filterType !== item.type) return;
          if (this.searchQuery && !item.id.toLowerCase().includes(this.searchQuery)) return;

          layers[layerKey].push({
            id: item.id,
            object: item.object,
            data: { type: item.type },
            name: item.id,
            visible: item.object.visible !== false
          });
          totalObjects++;
        }
      });
    }

    // Update object count
    document.getElementById("di-object-count").textContent = `${totalObjects} objects`;

    if (totalObjects === 0) {
      outliner.innerHTML = `<div class="di-empty">No objects found</div>`;
      return;
    }

    // Render layers
    Object.entries(layers).forEach(([layerKey, objects]) => {
      if (objects.length === 0) return;

      const layer = LAYER_CATEGORIES[layerKey];
      const layerDiv = this.createLayerElement(layerKey, layer, objects);
      outliner.appendChild(layerDiv);
    });
  }

  /**
   * Create a layer element with its objects
   */
  createLayerElement(layerKey, layer, objects) {
    const layerDiv = document.createElement("div");
    layerDiv.className = "di-layer";
    layerDiv.dataset.layer = layerKey;

    const isExpanded = this.layerVisibility[layerKey];

    layerDiv.innerHTML = `
      <div class="di-layer-header">
        <button class="di-layer-toggle-btn ${isExpanded ? "expanded" : ""}" title="Toggle layer">
          ${isExpanded ? "▼" : "▶"}
        </button>
        <div class="di-layer-icon" style="background: ${layer.color}">${layer.icon}</div>
        <span class="di-layer-name">${layer.name}</span>
        <span class="di-layer-count">${objects.length}</span>
      </div>
      <div class="di-layer-objects ${isExpanded ? "visible" : ""}"></div>
    `;

    const objectsContainer = layerDiv.querySelector(".di-layer-objects");
    const toggleBtn = layerDiv.querySelector(".di-layer-toggle-btn");
    const header = layerDiv.querySelector(".di-layer-header");

    // Toggle layer expansion
    toggleBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const content = layerDiv.querySelector(".di-layer-objects");
      content.classList.toggle("visible");
      toggleBtn.classList.toggle("expanded");
      toggleBtn.textContent = content.classList.contains("visible") ? "▼" : "▶";
    });

    // Also toggle on header click
    header.addEventListener("click", () => {
      toggleBtn.click();
    });

    // Add objects
    objects.forEach(obj => {
      const objEl = this.createObjectElement(obj);
      objectsContainer.appendChild(objEl);
    });

    return layerDiv;
  }

  /**
   * Create an object element in the outliner
   */
  createObjectElement(obj) {
    const div = document.createElement("div");
    div.className = "di-object-item";
    div.dataset.objectId = obj.id;
    if (this.selectedObjectId === obj.id) {
      div.classList.add("selected");
    }

    const isExpanded = this.expandedObjects.has(obj.id);
    const hasChildren = obj.object && obj.object.children && obj.object.children.length > 0;

    div.innerHTML = `
      <button class="di-expand-btn ${isExpanded ? "expanded" : ""}" style="visibility: ${hasChildren ? "visible" : "hidden"}">
        ▶
      </button>
      <span class="di-object-icon">${this.getObjectIcon(obj.data?.type)}</span>
      <span class="di-object-name">${obj.name}</span>
      <span class="di-object-type">${obj.data?.type || "obj"}</span>
      <button class="di-object-visible ${obj.visible === false ? "hidden" : ""}" title="Toggle visibility">
        ${obj.visible === false ? "👁️‍🗨️" : "👁️"}
      </button>
    `;

    // Click to select
    div.addEventListener("click", (e) => {
      if (e.target.closest(".di-expand-btn") || e.target.closest(".di-object-visible")) {
        return;
      }
      this.selectObject(obj.id);
    });

    // Toggle visibility
    const visBtn = div.querySelector(".di-object-visible");
    visBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      this.toggleObjectVisibility(obj);
    });

    // Expand button
    if (hasChildren) {
      const expandBtn = div.querySelector(".di-expand-btn");
      expandBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.toggleObjectExpansion(obj.id, div);
      });

      // Add children container
      if (isExpanded) {
        const children = this.createObjectChildren(obj.object);
        if (children) {
          div.appendChild(children);
        }
      }
    }

    return div;
  }

  /**
   * Create children element for an object
   */
  createObjectChildren(object) {
    if (!object.children || object.children.length === 0) return null;

    const container = document.createElement("div");
    container.className = "di-object-children visible";

    object.children.forEach((child, index) => {
      const childDiv = document.createElement("div");
      childDiv.className = "di-object-child";

      const icon = child.isMesh ? "🔷" :
                   child.isLight ? "💡" :
                   child.isCamera ? "📷" :
                   child.isGroup ? "📁" : "📄";

      childDiv.innerHTML = `
        <span class="di-child-icon">${icon}</span>
        <span>${child.name || child.type || `child_${index}`}</span>
      `;

      childDiv.addEventListener("click", (e) => {
        e.stopPropagation();
        // Could select child here
        this.setStatus(`Selected child: ${child.name || child.type}`);
      });

      container.appendChild(childDiv);
    });

    return container;
  }

  /**
   * Toggle object expansion in outliner
   */
  toggleObjectExpansion(id, element) {
    const existingChildren = element.querySelector(".di-object-children");

    if (existingChildren) {
      existingClasses.toggle("visible");
      if (!existingChildren.classList.contains("visible")) {
        this.expandedObjects.delete(id);
        element.querySelector(".di-expand-btn").classList.remove("expanded");
      }
    } else {
      // Create children
      const obj = this.getObjectById(id);
      if (obj) {
        const children = this.createObjectChildren(obj.object);
        if (children) {
          element.appendChild(children);
          this.expandedObjects.add(id);
          element.querySelector(".di-expand-btn").classList.add("expanded");
        }
      }
    }
  }

  /**
   * Toggle layer visibility
   */
  toggleLayer(layerKey) {
    this.layerVisibility[layerKey] = !this.layerVisibility[layerKey];

    // Update toggle button
    const btn = document.querySelector(`.di-layer-toggle[title*="${layerKey}"]`);
    if (btn) {
      btn.classList.toggle("active", this.layerVisibility[layerKey]);
      btn.classList.toggle("hidden", !this.layerVisibility[layerKey]);
    }

    // Update layer content visibility
    const layerDiv = document.querySelector(`.di-layer[data-layer="${layerKey}"]`);
    if (layerDiv) {
      const objects = layerDiv.querySelector(".di-layer-objects");
      if (objects) {
        objects.classList.toggle("visible", this.layerVisibility[layerKey]);
      }
    }

    // Update actual object visibility
    this.updateLayerObjectVisibility(layerKey);
    this.setStatus(`${this.layerVisibility[layerKey] ? "Showed" : "Hid"} ${LAYER_CATEGORIES[layerKey].name}`);
  }

  /**
   * Update visibility of objects in a layer
   */
  updateLayerObjectVisibility(layerKey) {
    const visible = this.layerVisibility[layerKey];

    if (layerKey === "splats" || layerKey === "gltf") {
      this.sceneManager?.objects.forEach((obj, id) => {
        const data = this.sceneManager.objectData.get(id);
        const type = data?.type;
        if ((layerKey === "splats" && type === "splat") ||
            (layerKey === "gltf" && type === "gltf")) {
          obj.visible = visible;
        }
      });
    } else if (layerKey === "lights" && window.lightManager) {
      window.lightManager.lights.forEach((light) => {
        light.visible = visible;
      });
    } else if (layerKey === "videos" && window.videoManager) {
      window.videoManager.videoPlayers.forEach((player) => {
        if (player.mesh) player.mesh.visible = visible;
      });
    }
  }

  /**
   * Toggle individual object visibility
   */
  toggleObjectVisibility(obj) {
    obj.visible = !obj.visible;
    obj.object.visible = obj.visible;

    // Update button
    const btn = document.querySelector(`.di-object-item[data-object-id="${obj.id}"] .di-object-visible`);
    if (btn) {
      btn.classList.toggle("hidden", !obj.visible);
      btn.textContent = obj.visible ? "👁️" : "👁️‍🗨️";
    }

    this.setStatus(`${obj.visible ? "Showed" : "Hid"} ${obj.name}`);
  }

  /**
   * Select an object
   */
  selectObject(id) {
    // Deselect previous
    if (this.selectedObjectId) {
      const prevEl = document.querySelector(`.di-object-item[data-object-id="${this.selectedObjectId}"]`);
      if (prevEl) prevEl.classList.remove("selected");
    }

    this.selectedObjectId = id;

    // Select new
    const newEl = document.querySelector(`.di-object-item[data-object-id="${id}"]`);
    if (newEl) newEl.classList.add("selected");

    // Update properties panel
    this.updatePropertiesPanel(id);

    // Register with gizmo if not already
    const obj = this.getObjectById(id);
    if (obj && this.gizmoManager) {
      const hasGizmo = this.gizmoManager.objects.some(o => o.id === id);
      if (!hasGizmo) {
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
      const prevEl = document.querySelector(`.di-object-item[data-object-id="${this.selectedObjectId}"]`);
      if (prevEl) prevEl.classList.remove("selected");
    }

    this.selectedObjectId = null;
    this.updatePropertiesPanel(null);
    this.setStatus("Deselected");
  }

  /**
   * Update the properties panel
   */
  updatePropertiesPanel(id) {
    const panel = document.getElementById("di-properties");
    if (!panel) return;

    if (!id) {
      panel.innerHTML = `<div class="di-empty">Select an object to view properties</div>`;
      return;
    }

    const obj = this.getObjectById(id);
    if (!obj) {
      panel.innerHTML = `<div class="di-empty">Object not found</div>`;
      return;
    }

    const o = obj.object;
    const data = obj.data || {};

    panel.innerHTML = `
      ${this.createTransformSection(id, o, data)}
      ${this.createVisibilitySection(id, o, data)}
      ${this.createInfoSection(id, o, data)}
      ${this.createActionsSection(id, o, data)}
    `;

    // Bind input events for live updates
    this.bindPropertyInputs(id, o);
  }

  /**
   * Create transform property section
   */
  createTransformSection(id, obj, data) {
    const pos = obj.position;
    const rot = obj.rotation;
    const scale = obj.scale;

    return `
      <div class="di-prop-section">
        <div class="di-prop-section-header" data-section="transform">
          <span class="di-prop-icon">📍</span>
          <span class="di-prop-title">Transform</span>
          <button class="di-prop-toggle">▼</button>
        </div>
        <div class="di-prop-section-content visible">
          <div class="di-prop-row">
            <span class="di-prop-label">Position</span>
            <div class="di-prop-inputs">
              <div class="di-prop-input-group x">
                <span class="di-prop-axis">X</span>
                <input type="number" class="di-prop-input di-prop-pos-x" step="0.1" value="${pos.x.toFixed(3)}" />
              </div>
              <div class="di-prop-input-group y">
                <span class="di-prop-axis">Y</span>
                <input type="number" class="di-prop-input di-prop-pos-y" step="0.1" value="${pos.y.toFixed(3)}" />
              </div>
              <div class="di-prop-input-group z">
                <span class="di-prop-axis">Z</span>
                <input type="number" class="di-prop-input di-prop-pos-z" step="0.1" value="${pos.z.toFixed(3)}" />
              </div>
            </div>
          </div>
          <div class="di-prop-row">
            <span class="di-prop-label">Rotation</span>
            <div class="di-prop-inputs">
              <div class="di-prop-input-group x">
                <span class="di-prop-axis">X</span>
                <input type="number" class="di-prop-input di-prop-rot-x" step="0.01" value="${rot.x.toFixed(4)}" />
              </div>
              <div class="di-prop-input-group y">
                <span class="di-prop-axis">Y</span>
                <input type="number" class="di-prop-input di-prop-rot-y" step="0.01" value="${rot.y.toFixed(4)}" />
              </div>
              <div class="di-prop-input-group z">
                <span class="di-prop-axis">Z</span>
                <input type="number" class="di-prop-input di-prop-rot-z" step="0.01" value="${rot.z.toFixed(4)}" />
              </div>
            </div>
          </div>
          <div class="di-prop-row">
            <span class="di-prop-label">Scale</span>
            <div class="di-prop-inputs">
              <div class="di-prop-input-group x">
                <span class="di-prop-axis">X</span>
                <input type="number" class="di-prop-input di-prop-scale-x" step="0.1" value="${scale.x.toFixed(3)}" />
              </div>
              <div class="di-prop-input-group y">
                <span class="di-prop-axis">Y</span>
                <input type="number" class="di-prop-input di-prop-scale-y" step="0.1" value="${scale.y.toFixed(3)}" />
              </div>
              <div class="di-prop-input-group z">
                <span class="di-prop-axis">Z</span>
                <input type="number" class="di-prop-input di-prop-scale-z" step="0.1" value="${scale.z.toFixed(3)}" />
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  /**
   * Create visibility section
   */
  createVisibilitySection(id, obj, data) {
    const isVisible = obj.visible !== false;

    return `
      <div class="di-prop-section">
        <div class="di-prop-section-header" data-section="visibility">
          <span class="di-prop-icon">👁️</span>
          <span class="di-prop-title">Visibility</span>
          <button class="di-prop-toggle">▼</button>
        </div>
        <div class="di-prop-section-content visible">
          <div class="di-prop-value">
            <span class="di-prop-value-label">Visible</span>
            <label style="display: flex; align-items: center; gap: 8px;">
              <input type="checkbox" class="di-checkbox di-prop-visible" ${isVisible ? "checked" : ""} />
              <span style="font-size: 11px;">${isVisible ? "Yes" : "No"}</span>
            </label>
          </div>
          <div class="di-prop-value">
            <span class="di-prop-value-label">Render Order</span>
            <span class="di-value-badge">${obj.renderOrder || 0}</span>
          </div>
          ${obj.castShadow !== undefined ? `
            <div class="di-prop-value">
              <span class="di-prop-value-label">Cast Shadow</span>
              <span class="di-value-badge">${obj.castShadow ? "Yes" : "No"}</span>
            </div>
          ` : ""}
          ${obj.receiveShadow !== undefined ? `
            <div class="di-prop-value">
              <span class="di-prop-value-label">Receive Shadow</span>
              <span class="di-value-badge">${obj.receiveShadow ? "Yes" : "No"}</span>
            </div>
          ` : ""}
        </div>
      </div>
    `;
  }

  /**
   * Create info section
   */
  createInfoSection(id, obj, data) {
    const childCount = obj.children ? obj.children.length : 0;

    return `
      <div class="di-prop-section">
        <div class="di-prop-section-header" data-section="info">
          <span class="di-prop-icon">ℹ️</span>
          <span class="di-prop-title">Info</span>
          <button class="di-prop-toggle">▼</button>
        </div>
        <div class="di-prop-section-content visible">
          <div class="di-prop-value">
            <span class="di-prop-value-label">ID</span>
            <span class="di-value-badge" style="font-family: monospace;">${id}</span>
          </div>
          <div class="di-prop-value">
            <span class="di-prop-value-label">Type</span>
            <span class="di-value-badge">${data.type || obj.type || "Object"}</span>
          </div>
          <div class="di-prop-value">
            <span class="di-prop-value-label">Children</span>
            <span class="di-value-badge">${childCount}</span>
          </div>
          ${data.description ? `
            <div class="di-prop-value">
              <span class="di-prop-value-label">Description</span>
              <span style="font-size: 11px; color: #888; max-width: 180px; overflow: hidden; text-overflow: ellipsis;">${data.description}</span>
            </div>
          ` : ""}
          ${data.path ? `
            <div class="di-prop-value">
              <span class="di-prop-value-label">Path</span>
              <span class="di-value-badge" style="font-family: monospace; font-size: 9px;">${data.path.split("/").pop()}</span>
            </div>
          ` : ""}
        </div>
      </div>
    `;
  }

  /**
   * Create actions section
   */
  createActionsSection(id, obj, data) {
    return `
      <div class="di-prop-section">
        <div class="di-prop-section-header" data-section="actions">
          <span class="di-prop-icon">⚡</span>
          <span class="di-prop-title">Actions</span>
          <button class="di-prop-toggle">▼</button>
        </div>
        <div class="di-prop-section-content visible">
          <div class="di-prop-actions">
            <button class="di-action-btn primary" id="di-focus-btn">
              <span>🎯</span> Focus
            </button>
            <button class="di-action-btn secondary" id="di-copy-btn">
              <span>📋</span> Copy
            </button>
          </div>
          <div class="di-prop-actions">
            <button class="di-action-btn secondary" id="di-gizmo-btn">
              <span>🔧</span> Gizmo
            </button>
            <button class="di-action-btn danger" id="di-delete-btn">
              <span>🗑️</span> Delete
            </button>
          </div>
        </div>
      </div>
    `;
  }

  /**
   * Bind property input events for live updates
   */
  bindPropertyInputs(id, obj) {
    // Position inputs
    ["x", "y", "z"].forEach(axis => {
      const input = document.querySelector(`.di-prop-pos-${axis}`);
      if (input) {
        input.addEventListener("input", (e) => {
          const value = parseFloat(e.target.value);
          if (!isNaN(value)) {
            obj.position[axis] = value;
            this.setStatus(`Position.${axis.toUpperCase()} = ${value}`);
          }
        });
      }
    });

    // Rotation inputs
    ["x", "y", "z"].forEach(axis => {
      const input = document.querySelector(`.di-prop-rot-${axis}`);
      if (input) {
        input.addEventListener("input", (e) => {
          const value = parseFloat(e.target.value);
          if (!isNaN(value)) {
            obj.rotation[axis] = value;
            obj.updateMatrix();
            this.setStatus(`Rotation.${axis.toUpperCase()} = ${value}`);
          }
        });
      }
    });

    // Scale inputs
    ["x", "y", "z"].forEach(axis => {
      const input = document.querySelector(`.di-prop-scale-${axis}`);
      if (input) {
        input.addEventListener("input", (e) => {
          const value = parseFloat(e.target.value);
          if (!isNaN(value)) {
            obj.scale[axis] = value;
            this.setStatus(`Scale.${axis.toUpperCase()} = ${value}`);
          }
        });
      }
    });

    // Visibility checkbox
    const visCheckbox = document.querySelector(".di-prop-visible");
    if (visCheckbox) {
      visCheckbox.addEventListener("change", (e) => {
        obj.visible = e.target.checked;
        this.setStatus(`Visible = ${e.target.checked}`);
      });
    }

    // Focus button
    const focusBtn = document.getElementById("di-focus-btn");
    if (focusBtn) {
      focusBtn.addEventListener("click", () => this.focusOnSelected());
    }

    // Copy button
    const copyBtn = document.getElementById("di-copy-btn");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => this.copyTransform(obj));
    }

    // Gizmo button
    const gizmoBtn = document.getElementById("di-gizmo-btn");
    if (gizmoBtn) {
      gizmoBtn.addEventListener("click", () => this.toggleGizmo(id));
    }

    // Delete button
    const deleteBtn = document.getElementById("di-delete-btn");
    if (deleteBtn) {
      deleteBtn.addEventListener("click", () => this.deleteSelectedObject());
    }

    // Section toggles
    document.querySelectorAll(".di-prop-section-header").forEach(header => {
      header.addEventListener("click", () => {
        const content = header.nextElementSibling;
        const toggle = header.querySelector(".di-prop-toggle");
        content.classList.toggle("visible");
        toggle.textContent = content.classList.contains("visible") ? "▼" : "▶";
      });
    });
  }

  /**
   * Focus camera on selected object
   */
  focusOnSelected() {
    if (!this.selectedObjectId || !this.characterController) return;

    const obj = this.getObjectById(this.selectedObjectId);
    if (!obj) return;

    const targetPos = obj.object.position.clone();
    const offset = new THREE.Vector3(0, 0, 5);

    // Move character to look at object
    if (this.characterController.character) {
      this.characterController.character.setTranslation(
        {
          x: targetPos.x,
          y: Math.max(0.9, targetPos.y),
          z: targetPos.z + 5
        },
        true
      );
    }

    this.setStatus(`Focused on ${this.selectedObjectId}`);
  }

  /**
   * Copy object transform to clipboard
   */
  copyTransform(obj) {
    const transform = {
      position: {
        x: obj.position.x.toFixed(2),
        y: obj.position.y.toFixed(2),
        z: obj.position.z.toFixed(2)
      },
      rotation: {
        x: obj.rotation.x.toFixed(4),
        y: obj.rotation.y.toFixed(4),
        z: obj.rotation.z.toFixed(4)
      },
      scale: {
        x: obj.scale.x.toFixed(2),
        y: obj.scale.y.toFixed(2),
        z: obj.scale.z.toFixed(2)
      }
    };

    const text = JSON.stringify(transform, null, 2);
    navigator.clipboard.writeText(text).then(() => {
      this.setStatus("Copied transform to clipboard");
    });

    // Also log to console for easy access
    console.log("Transform:", transform);
  }

  /**
   * Toggle gizmo for selected object
   */
  toggleGizmo(id) {
    if (!this.gizmoManager) return;

    const obj = this.getObjectById(id);
    if (!obj) return;

    const hasGizmo = this.gizmoManager.objects.some(o => o.id === id);

    if (hasGizmo) {
      this.gizmoManager.unregisterObject(obj.object);
      this.setStatus(`Removed gizmo from ${id}`);
    } else {
      this.gizmoManager.registerObject(obj.object, id, obj.data?.type || "object");
      this.setStatus(`Added gizmo to ${id}`);
    }

    this.refreshOutliner();
  }

  /**
   * Delete selected object
   */
  deleteSelectedObject() {
    if (!this.selectedObjectId) return;

    if (!confirm(`Delete "${this.selectedObjectId}"?`)) return;

    if (this.sceneManager && this.sceneManager.hasObject(this.selectedObjectId)) {
      this.sceneManager.removeObject(this.selectedObjectId);
    } else {
      const obj = this.getObjectById(this.selectedObjectId);
      if (obj && obj.object.parent) {
        obj.object.parent.remove(obj.object);
      }
    }

    this.deselectObject();
    this.refreshOutliner();
    this.setStatus(`Deleted ${this.selectedObjectId}`);
  }

  /**
   * Get icon for object type
   */
  getObjectIcon(type) {
    const icons = {
      splat: "🌐",
      gltf: "📦",
      light: "💡",
      video: "🎬",
      collider: "🔷",
      spawned: "✨"
    };
    return icons[type] || "📄";
  }

  /**
   * Set status message
   */
  setStatus(message) {
    const status = document.getElementById("di-status");
    if (status) {
      status.textContent = message;
      setTimeout(() => {
        if (status.textContent === message) {
          status.textContent = "Ready";
        }
      }, 3000);
    }
  }

  /**
   * Open the inspector
   */
  open() {
    if (this.isOpen) return;

    this.isOpen = true;
    this.element.classList.remove("hidden");

    // Release pointer lock
    if (document.pointerLockElement) {
      document.exitPointerLock();
    }

    // Refresh outliner when opened
    this.refreshOutliner();
    this.setStatus("Inspector opened");
  }

  /**
   * Close the inspector
   */
  close() {
    if (!this.isOpen) return;

    this.isOpen = false;
    this.element.classList.add("hidden");
    this.setStatus("Inspector closed");
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

    // Remove styles
    const styles = document.getElementById("debug-inspector-styles");
    if (styles && styles.parentNode) {
      styles.parentNode.removeChild(styles);
    }

    if (window.debugInspector === this) {
      delete window.debugInspector;
    }
  }
}

export default DebugInspector;
