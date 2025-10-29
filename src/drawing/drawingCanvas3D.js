import * as THREE from "three";

const CANVAS_SIZE = 500;
const STROKE_WEIGHT = 3;

export class DrawingCanvas3D {
  constructor(scene, position = { x: 0, y: 1.5, z: -2 }, scale = 1) {
    this.scene = scene;
    this.canvas = document.createElement("canvas");
    this.canvas.width = CANVAS_SIZE;
    this.canvas.height = CANVAS_SIZE;
    this.ctx = this.canvas.getContext("2d");

    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.needsUpdate = true;

    const geometry = new THREE.PlaneGeometry(scale, scale);
    const material = new THREE.MeshBasicMaterial({
      map: this.texture,
      side: THREE.DoubleSide,
    });

    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.position.set(position.x, position.y, position.z);
    this.mesh.userData.isDrawingCanvas = true;

    this.scene.add(this.mesh);

    this.isDrawing = false;
    this.currentStroke = [[], []];
    this.imageStrokes = [];
    this.lastPoint = null;

    this.clearCanvas();
  }

  clearCanvas() {
    this.ctx.fillStyle = "#FFFFFF";
    this.ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    this.ctx.strokeStyle = "black";
    this.ctx.lineWidth = STROKE_WEIGHT;
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";
    this.texture.needsUpdate = true;
    this.imageStrokes = [];
    this.currentStroke = [[], []];
    this.lastPoint = null;
  }

  startStroke(uv) {
    if (!uv) return;

    this.isDrawing = true;
    const x = Math.floor(uv.x * CANVAS_SIZE);
    const y = Math.floor((1 - uv.y) * CANVAS_SIZE);

    this.lastPoint = { x, y };
    this.currentStroke = [[x], [y]];
  }

  addPoint(uv) {
    if (!this.isDrawing || !uv || !this.lastPoint) return;

    const x = Math.floor(uv.x * CANVAS_SIZE);
    const y = Math.floor((1 - uv.y) * CANVAS_SIZE);

    if (x < 0 || x >= CANVAS_SIZE || y < 0 || y >= CANVAS_SIZE) return;

    // Interpolate points for smooth lines (needed for preprocessing)
    const dx = x - this.lastPoint.x;
    const dy = y - this.lastPoint.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const steps = Math.max(Math.floor(distance), 1);

    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      const ix = Math.floor(this.lastPoint.x + dx * t);
      const iy = Math.floor(this.lastPoint.y + dy * t);
      this.currentStroke[0].push(ix);
      this.currentStroke[1].push(iy);
    }

    this.ctx.beginPath();
    this.ctx.moveTo(this.lastPoint.x, this.lastPoint.y);
    this.ctx.lineTo(x, y);
    this.ctx.stroke();

    this.lastPoint = { x, y };
    this.texture.needsUpdate = true;
  }

  endStroke() {
    if (this.isDrawing && this.currentStroke[0].length > 0) {
      // Deep copy the stroke arrays
      this.imageStrokes.push([
        [...this.currentStroke[0]],
        [...this.currentStroke[1]],
      ]);
    }

    this.isDrawing = false;
    this.currentStroke = [[], []];
    this.lastPoint = null;
  }

  getStrokes() {
    return this.imageStrokes;
  }

  hasStrokes() {
    return this.imageStrokes.length > 0;
  }

  getMesh() {
    return this.mesh;
  }

  dispose() {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh.material.dispose();
      this.texture.dispose();
    }
  }

  update() {
    if (this.texture.needsUpdate) {
      this.texture.needsUpdate = true;
    }
  }
}
