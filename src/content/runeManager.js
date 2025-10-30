import { StrokeMesh } from "../vfx/strokeMesh.js";
import * as THREE from "three";
import { Logger } from "../utils/logger.js";
import { STROKE_SHAPES } from "../drawing/strokeData.js";

export class RuneManager {
  constructor(scene) {
    this.scene = scene;
    this.logger = new Logger("RuneManager", true);
    this.runes = [];
    this.isActive = false;
  }

  createRune(type, position, options = {}) {
    let runeData;

    // If custom stroke data provided, use it
    if (options.strokeData) {
      runeData = options.strokeData;
    } else {
      runeData = STROKE_SHAPES[type];
      if (!runeData) {
        this.logger.warn(`Unknown rune type: ${type}`);
        return null;
      }
    }

    const scale = options.scale || 0.3;
    const color = options.color || new THREE.Color(0xaaeeff);
    const rotation = options.rotation || { x: 0, y: 0, z: 0 };

    const strokeMesh = new StrokeMesh(this.scene, {
      scale,
      color,
      position,
      rotation,
      renderOrder: 10001,
      isStatic: true,
    });

    strokeMesh.setStrokeData(runeData);

    this.runes.push({
      type: type || "custom",
      mesh: strokeMesh,
      position,
    });

    this.isActive = true;
    this.logger.log(`Created ${type || "custom"} rune at`, position);
    return strokeMesh;
  }

  spawnRunesAroundPosition(centerPosition, radius = 2, count = 3) {
    const types = ["lightning", "star", "circle"];
    const angleStep = (Math.PI * 2) / count;

    for (let i = 0; i < count; i++) {
      const angle = angleStep * i + Math.random() * 0.3 - 0.15;
      const distance = radius + Math.random() * 0.5;

      const position = {
        x: centerPosition.x + Math.cos(angle) * distance,
        y: centerPosition.y + (Math.random() - 0.5) * 0.5,
        z: centerPosition.z + Math.sin(angle) * distance,
      };

      const rotation = {
        x: 0,
        y: Math.random() * Math.PI * 2,
        z: 0,
      };

      const type = types[i % types.length];
      this.createRune(type, position, {
        scale: 0.25 + Math.random() * 0.15,
        rotation,
      });
    }

    this.isActive = true;
    this.logger.log(`Spawned ${count} runes around`, centerPosition);
  }

  clearRunes() {
    for (const rune of this.runes) {
      rune.mesh.dispose();
    }
    this.runes = [];
    this.isActive = false;
    this.logger.log("Cleared all runes");
  }

  update(dt) {
    if (!this.isActive) return;

    for (const rune of this.runes) {
      rune.mesh.update(dt);
    }
  }

  dispose() {
    this.clearRunes();
  }
}
