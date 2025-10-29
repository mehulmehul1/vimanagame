import * as THREE from "three";

const CANVAS_SIZE = 500;
const STROKE_WEIGHT = 3;
const PARTICLE_GRID_DENSITY = 100; // Particles per side (100x100 = 10000 particles, ~2x original)

export class ParticleCanvas3D {
  constructor(
    scene,
    position = { x: 0, y: 1.5, z: -2 },
    scale = 1,
    enableParticles = true
  ) {
    this.scene = scene;
    this.enableParticles = enableParticles;

    // Hidden 2D canvas for stroke recording (ML recognition)
    this.canvas = document.createElement("canvas");
    this.canvas.width = CANVAS_SIZE;
    this.canvas.height = CANVAS_SIZE;
    this.ctx = this.canvas.getContext("2d");

    // Touch texture canvas (for particle displacement)
    this.touchCanvas = document.createElement("canvas");
    this.touchCanvas.width = PARTICLE_GRID_DENSITY;
    this.touchCanvas.height = PARTICLE_GRID_DENSITY;
    this.touchCtx = this.touchCanvas.getContext("2d");

    // Touch texture with proper filtering
    this.touchTexture = new THREE.CanvasTexture(this.touchCanvas);
    this.touchTexture.minFilter = THREE.LinearFilter;
    this.touchTexture.magFilter = THREE.LinearFilter;
    this.touchTexture.needsUpdate = true;

    // Create visible white plane (for drawing visibility)
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.needsUpdate = true;

    const geometry = new THREE.PlaneGeometry(scale, scale);
    const material = new THREE.MeshBasicMaterial({
      map: this.texture,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 1.0,
      alphaTest: 0.01, // Only render where there's actual drawing
      depthWrite: false, // Allow proper blending on top of splats
    });

    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.position.set(position.x, position.y, position.z);
    this.mesh.userData.isDrawingCanvas = true;
    this.mesh.renderOrder = 9999; // Render on top of splats (9998)
    this.scene.add(this.mesh);

    // Conditionally create particle system (in front of the plane)
    this.particleSystem = null;
    this.particleOffset = 0.05;

    if (this.enableParticles) {
      this.particleSystem = this.createParticleSystem(scale);
      this.particleSystem.position.set(position.x, position.y, position.z);
      this.particleSystem.renderOrder = 9999; // Render on top of splats (9998)
      this.scene.add(this.particleSystem);
    }

    this.isDrawing = false;
    this.currentStroke = [[], []];
    this.lastPoint = null;
    this.time = 0;

    // Track stroke segments with timestamps for progressive fade
    // ML strokes are dynamically built from active segments
    this.strokeSegments = []; // Array of {x1, y1, x2, y2, timestamp, rapidFade, strokeIndex}
    this.fadeDuration = 6.0; // Seconds for stroke to fade
    this.rapidFadeDuration = 0.8; // Rapid fade when clearing (800ms)
    this.currentStrokeIndex = 0; // Counter for linking segments to strokes

    this.clearCanvas();
  }

  createParticleSystem(scale) {
    const numParticles = PARTICLE_GRID_DENSITY * PARTICLE_GRID_DENSITY;

    // Create instanced geometry
    const geometry = new THREE.InstancedBufferGeometry();

    // Base quad geometry (4 vertices, 2 triangles)
    const positions = new THREE.BufferAttribute(new Float32Array(4 * 3), 3);
    positions.setXYZ(0, -0.5, 0.5, 0.0);
    positions.setXYZ(1, 0.5, 0.5, 0.0);
    positions.setXYZ(2, -0.5, -0.5, 0.0);
    positions.setXYZ(3, 0.5, -0.5, 0.0);
    geometry.setAttribute("position", positions);

    // UVs for each particle quad
    const uvs = new THREE.BufferAttribute(new Float32Array(4 * 2), 2);
    uvs.setXY(0, 0.0, 0.0);
    uvs.setXY(1, 1.0, 0.0);
    uvs.setXY(2, 0.0, 1.0);
    uvs.setXY(3, 1.0, 1.0);
    geometry.setAttribute("uv", uvs);

    // Triangle indices
    geometry.setIndex(
      new THREE.BufferAttribute(new Uint16Array([0, 2, 1, 2, 3, 1]), 1)
    );

    // Instance attributes
    const indices = new Uint16Array(numParticles);
    const offsets = new Float32Array(numParticles * 3);
    const angles = new Float32Array(numParticles);
    const velocities = new Float32Array(numParticles * 3);
    const sizes = new Float32Array(numParticles);

    // Position particles in a grid with organic edge variation
    for (let i = 0; i < numParticles; i++) {
      const x = (i % PARTICLE_GRID_DENSITY) / PARTICLE_GRID_DENSITY;
      const y = Math.floor(i / PARTICLE_GRID_DENSITY) / PARTICLE_GRID_DENSITY;

      // Center the grid
      const centeredX = x - 0.5;
      const centeredY = y - 0.5;

      // Calculate distance from center (0 at center, 1 at corners)
      const distFromCenter =
        Math.sqrt(centeredX * centeredX + centeredY * centeredY) * Math.sqrt(2);

      // Calculate distance from edges (0 at edges, 1 at center)
      const distFromEdgeX = 1.0 - Math.abs(centeredX) * 2.0;
      const distFromEdgeY = 1.0 - Math.abs(centeredY) * 2.0;
      const distFromEdge = Math.min(distFromEdgeX, distFromEdgeY);

      // Edge decay factor: 0 at center, increases toward edges
      // Use exponential curve for more dramatic edge variation
      const edgeDecay = Math.pow(1.0 - distFromEdge, 2.5);

      // Add organic variation that increases dramatically at edges
      const edgeVariationStrength = edgeDecay * 0.15; // Max 15% of scale
      const noiseX =
        (Math.random() - 0.5) * 2.0 * edgeVariationStrength * scale;
      const noiseY =
        (Math.random() - 0.5) * 2.0 * edgeVariationStrength * scale;
      const noiseZ =
        (Math.random() - 0.5) * edgeVariationStrength * scale * 0.3;

      offsets[i * 3 + 0] = centeredX * scale + noiseX;
      offsets[i * 3 + 1] = centeredY * scale + noiseY;
      offsets[i * 3 + 2] = noiseZ;

      indices[i] = i;
      angles[i] = Math.random() * Math.PI * 2;

      // Random gentle velocities for floating motion
      velocities[i * 3 + 0] = (Math.random() - 0.5) * 10.0;
      velocities[i * 3 + 1] = (Math.random() - 0.5) * 10.0;
      velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.5;

      // Random size between 0.25 and 1.0, with smaller particles at edges
      const sizeVariation = 0.25 + Math.random() * 0.75;
      sizes[i] = sizeVariation * (1.0 - edgeDecay * 0.3); // Reduce size at edges by up to 30%
    }

    geometry.setAttribute(
      "pindex",
      new THREE.InstancedBufferAttribute(indices, 1, false)
    );
    geometry.setAttribute(
      "offset",
      new THREE.InstancedBufferAttribute(offsets, 3, false)
    );
    geometry.setAttribute(
      "angle",
      new THREE.InstancedBufferAttribute(angles, 1, false)
    );
    geometry.setAttribute(
      "velocity",
      new THREE.InstancedBufferAttribute(velocities, 3, false)
    );
    geometry.setAttribute(
      "particleSize",
      new THREE.InstancedBufferAttribute(sizes, 1, false)
    );

    // Shader material
    const material = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uSize: { value: 0.015 * scale }, // Scaled for denser particles
        uTouch: { value: this.touchTexture },
        uColor: { value: new THREE.Color(0xaaeeff) }, // Bright whitish-blue glow
      },
      vertexShader: `
        attribute float pindex;
        attribute vec3 offset;
        attribute float angle;
        attribute vec3 velocity;
        attribute float particleSize;
        
        uniform float uTime;
        uniform float uSize;
        uniform sampler2D uTouch;
        
        varying vec2 vUv;
        varying float vDisplacement;
        
        // Simple hash function for noise
        float hash(float n) {
          return fract(sin(n) * 43758.5453123);
        }
        
        // Smooth noise
        float noise(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          
          float n = i.x + i.y * 57.0;
          return mix(
            mix(hash(n + 0.0), hash(n + 1.0), f.x),
            mix(hash(n + 57.0), hash(n + 58.0), f.x),
            f.y
          );
        }
        
        void main() {
          vUv = uv;
          
          // Get particle position in UV space (0-1)
          vec2 puv = offset.xy + 0.5;
          
          // Sample touch texture
          float touch = texture2D(uTouch, puv).r;
          
          // Base position with floating motion
          vec3 displaced = offset;
          
          // Add gentle drifting motion using noise
          float timeScale = uTime * 0.5;
          vec2 noiseCoord = vec2(pindex * 0.01, timeScale * 0.1);
          float drift = noise(noiseCoord) - 0.5;
          
          // Apply velocity-based drift (circular motion)
          // But reduce drift in areas where touch is present
          float driftAmount = 0.005 * (1.0 - touch * 0.8);
          displaced.x += sin(timeScale + pindex) * velocity.x * driftAmount;
          displaced.y += cos(timeScale + pindex * 1.3) * velocity.y * driftAmount;
          displaced.z += drift * velocity.z * driftAmount * 0.5;
          
          // Continuous avoidance force - particles flee from strokes with smooth momentum
          // Apply easing to touch for smoother acceleration/deceleration
          float touchSmooth = smoothstep(0.0, 1.0, touch); // Smooth curve
          touchSmooth = touchSmooth * touchSmooth; // Quadratic easing for gentler motion
          
          // Displacement with momentum (slower and smoother)
          float displacementStrength = 0.12; // Reduced for gentler movement
          float displacementZ = touchSmooth * displacementStrength;
          displaced.z += displacementZ;
          
          // Lateral spread with smooth easing
          float spreadAmount = 0.15; // Reduced spread
          float spreadEased = touchSmooth * spreadAmount;
          displaced.x += cos(angle) * spreadEased;
          displaced.y += sin(angle) * spreadEased;
          
          vDisplacement = touch;
          
          // Transform to world space
          vec4 mvPosition = modelViewMatrix * vec4(displaced, 1.0);
          // Apply random size variation (between 1/4 and full size)
          mvPosition.xyz += position * uSize * particleSize;
          
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        varying vec2 vUv;
        varying float vDisplacement;
        
         void main() {
           // Circular particle shape with soft glow
           float dist = distance(vUv, vec2(0.5));
           float alpha = 1.0 - smoothstep(0.2, 0.5, dist);
           
           // Brighter glow, enhanced when displaced
           vec3 color = uColor;
           float brightness = 1.2 + vDisplacement * 0.3;
           color *= brightness;
           
           // Soft glow falloff
           alpha *= alpha; // Square for softer edges
           
           gl_FragColor = vec4(color, alpha);
         }
      `,
      transparent: true,
      depthTest: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending, // Additive for glowing effect
    });

    return new THREE.Mesh(geometry, material);
  }

  clearCanvas() {
    // If no segments exist yet (initial setup), clear canvases immediately
    if (this.strokeSegments.length === 0) {
      this.ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
      this.touchCtx.fillStyle = "rgba(0, 0, 0, 1)";
      this.touchCtx.fillRect(
        0,
        0,
        PARTICLE_GRID_DENSITY,
        PARTICLE_GRID_DENSITY
      );
      if (this.texture) this.texture.needsUpdate = true;
      this.touchTexture.needsUpdate = true;
    } else {
      // Mark all existing stroke segments for rapid fade instead of instant clear
      // This keeps particles animating smoothly back to position
      for (let segment of this.strokeSegments) {
        if (!segment.rapidFade) {
          segment.rapidFade = true;
          segment.rapidFadeStartTime = this.time;
          segment.rapidFadeStartAlpha =
            1.0 - (this.time - segment.timestamp) / this.fadeDuration;
        }
      }
    }

    // Clear current stroke data
    this.currentStroke = [[], []];
    this.lastPoint = null;

    // Note: strokeSegments will be removed naturally as they fade out
    // ML strokes are built dynamically from active segments via getStrokes()
  }

  startStroke(uv) {
    if (!uv) return;

    this.isDrawing = true;

    // Record on high-res canvas for ML
    const x = Math.floor(uv.x * CANVAS_SIZE);
    const y = Math.floor((1 - uv.y) * CANVAS_SIZE);

    this.lastPoint = { x, y };
    this.currentStroke = [[x], [y]];

    // Draw to touch texture
    this.drawToTouchTexture(uv.x, 1 - uv.y);
  }

  addPoint(uv) {
    if (!this.isDrawing || !uv || !this.lastPoint) return;

    const x = Math.floor(uv.x * CANVAS_SIZE);
    const y = Math.floor((1 - uv.y) * CANVAS_SIZE);

    if (x < 0 || x >= CANVAS_SIZE || y < 0 || y >= CANVAS_SIZE) return;

    // Interpolate points for smooth lines (ML recording)
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

    // Record stroke segment with timestamp for progressive fade
    this.strokeSegments.push({
      x1: this.lastPoint.x,
      y1: this.lastPoint.y,
      x2: x,
      y2: y,
      uvX: uv.x,
      uvY: 1 - uv.y,
      timestamp: this.time,
      rapidFade: false,
      strokeIndex: this.currentStrokeIndex, // Link segment to its parent stroke
    });

    // Draw to high-res canvas for ML
    this.ctx.beginPath();
    this.ctx.moveTo(this.lastPoint.x, this.lastPoint.y);
    this.ctx.lineTo(x, y);
    this.ctx.stroke();

    // Update visible plane texture
    if (this.texture) {
      this.texture.needsUpdate = true;
    }

    this.lastPoint = { x, y };
  }

  drawToTouchTexture(x, y) {
    const touchX = x * PARTICLE_GRID_DENSITY;
    const touchY = y * PARTICLE_GRID_DENSITY;

    // Draw a bright circle at touch point (smaller radius for tighter displacement)
    const radius = 1.5;

    // Use lighter blending to ensure full coverage
    this.touchCtx.globalCompositeOperation = "lighter";

    const gradient = this.touchCtx.createRadialGradient(
      touchX,
      touchY,
      0,
      touchX,
      touchY,
      radius
    );
    gradient.addColorStop(0, "rgba(255, 255, 255, 1)");
    gradient.addColorStop(0.5, "rgba(255, 255, 255, 1)"); // Stronger center
    gradient.addColorStop(1, "rgba(255, 255, 255, 0)");

    this.touchCtx.fillStyle = gradient;
    const size = radius * 2;
    this.touchCtx.fillRect(touchX - radius, touchY - radius, size, size);

    // Reset composite operation
    this.touchCtx.globalCompositeOperation = "source-over";

    this.touchTexture.needsUpdate = true;
  }

  endStroke() {
    if (this.isDrawing && this.currentStroke[0].length > 0) {
      // Increment stroke index for next stroke
      // (segments are already added to strokeSegments with current index)
      this.currentStrokeIndex++;
    }

    this.isDrawing = false;
    this.currentStroke = [[], []];
    this.lastPoint = null;
  }

  getStrokes() {
    // Dynamically build strokes from only the active (non-expired) segments
    // This ensures that old parts of long strokes are excluded from ML recognition

    if (this.strokeSegments.length === 0) {
      return [];
    }

    // Group active segments by stroke index and rebuild strokes
    const strokeMap = new Map();

    for (const segment of this.strokeSegments) {
      // Check if this segment is expired
      let isExpired = false;

      if (segment.rapidFade) {
        const rapidAge = this.time - segment.rapidFadeStartTime;
        isExpired = rapidAge > this.rapidFadeDuration;
      } else {
        const age = this.time - segment.timestamp;
        isExpired = age > this.fadeDuration;
      }

      // Only include non-expired segments
      if (!isExpired) {
        if (!strokeMap.has(segment.strokeIndex)) {
          strokeMap.set(segment.strokeIndex, {
            xPoints: [],
            yPoints: [],
            segments: [],
          });
        }
        strokeMap.get(segment.strokeIndex).segments.push(segment);
      }
    }

    // Rebuild stroke arrays from active segments
    const activeStrokes = [];

    for (const [strokeIndex, strokeData] of strokeMap.entries()) {
      const xPoints = [];
      const yPoints = [];

      // Sort segments by timestamp to maintain drawing order
      strokeData.segments.sort((a, b) => a.timestamp - b.timestamp);

      // Add all points from segments
      for (const segment of strokeData.segments) {
        // Add the start point (unless it's a duplicate of the previous end point)
        if (
          xPoints.length === 0 ||
          xPoints[xPoints.length - 1] !== segment.x1 ||
          yPoints[yPoints.length - 1] !== segment.y1
        ) {
          xPoints.push(segment.x1);
          yPoints.push(segment.y1);
        }
        // Always add the end point
        xPoints.push(segment.x2);
        yPoints.push(segment.y2);
      }

      // Only include strokes with at least 2 points
      if (xPoints.length >= 2) {
        activeStrokes.push([xPoints, yPoints]);
      }
    }

    return activeStrokes;
  }

  hasStrokes() {
    // Check if there are any non-expired segments
    const currentTime = this.time;

    for (const segment of this.strokeSegments) {
      let isExpired = false;

      if (segment.rapidFade) {
        const rapidAge = currentTime - segment.rapidFadeStartTime;
        isExpired = rapidAge > this.rapidFadeDuration;
      } else {
        const age = currentTime - segment.timestamp;
        isExpired = age > this.fadeDuration;
      }

      if (!isExpired) {
        return true;
      }
    }

    return false;
  }

  getMesh() {
    return this.mesh;
  }

  dispose() {
    // Dispose visible plane
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh.material.dispose();
      if (this.texture) {
        this.texture.dispose();
      }
    }

    // Dispose particle system
    if (this.particleSystem) {
      this.scene.remove(this.particleSystem);
      this.particleSystem.geometry.dispose();
      this.particleSystem.material.dispose();
    }

    // Dispose touch texture
    if (this.touchTexture) {
      this.touchTexture.dispose();
    }
  }

  update(dt = 0.016) {
    this.time += dt;

    // Update shader time
    if (this.particleSystem && this.particleSystem.material) {
      this.particleSystem.material.uniforms.uTime.value = this.time;
    }

    // Progressive fade: redraw strokes with age-based alpha
    if (this.strokeSegments.length > 0) {
      // Clear both canvases
      this.ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
      this.touchCtx.fillStyle = "rgba(0, 0, 0, 1)";
      this.touchCtx.fillRect(
        0,
        0,
        PARTICLE_GRID_DENSITY,
        PARTICLE_GRID_DENSITY
      );

      // Filter out expired segments and redraw active ones
      this.strokeSegments = this.strokeSegments.filter((segment) => {
        let alpha;

        // Handle rapid fade (triggered by clear)
        if (segment.rapidFade) {
          const rapidAge = this.time - segment.rapidFadeStartTime;

          if (rapidAge > this.rapidFadeDuration) {
            return false; // Remove expired rapid fade segment
          }

          // Fade from current alpha to 0 over rapidFadeDuration
          const rapidProgress = rapidAge / this.rapidFadeDuration;
          alpha = segment.rapidFadeStartAlpha * (1.0 - rapidProgress);
        } else {
          // Normal progressive fade
          const age = this.time - segment.timestamp;

          if (age > this.fadeDuration) {
            return false; // Remove expired segment
          }

          // Calculate alpha based on age (fade from 1.0 to 0.0)
          alpha = 1.0 - age / this.fadeDuration;
        }

        // Draw to visible canvas
        this.ctx.strokeStyle = `rgba(0, 255, 255, ${alpha})`;
        this.ctx.lineWidth = STROKE_WEIGHT;
        this.ctx.lineCap = "round";
        this.ctx.beginPath();
        this.ctx.moveTo(segment.x1, segment.y1);
        this.ctx.lineTo(segment.x2, segment.y2);
        this.ctx.stroke();

        // Calculate separate alpha for particle displacement
        // Keep particles displaced longer - only start returning in last 30% of fade
        let touchAlpha;
        if (segment.rapidFade) {
          touchAlpha = alpha; // Rapid fade uses same curve
        } else {
          const age = this.time - segment.timestamp;
          const progress = age / this.fadeDuration;

          if (progress < 0.7) {
            // Stay at full displacement for first 70% of fade time
            touchAlpha = 1.0;
          } else {
            // Quick fade in the last 30%
            const fadeProgress = (progress - 0.7) / 0.3;
            touchAlpha = 1.0 - fadeProgress;
          }
        }

        // Draw to touch texture for particle displacement
        const touchX = segment.uvX * PARTICLE_GRID_DENSITY;
        const touchY = segment.uvY * PARTICLE_GRID_DENSITY;

        const radius = 1.5;
        this.touchCtx.globalCompositeOperation = "lighter";
        const gradient = this.touchCtx.createRadialGradient(
          touchX,
          touchY,
          0,
          touchX,
          touchY,
          radius
        );
        gradient.addColorStop(0, `rgba(255, 255, 255, ${touchAlpha})`);
        gradient.addColorStop(0.5, `rgba(255, 255, 255, ${touchAlpha})`);
        gradient.addColorStop(1, "rgba(255, 255, 255, 0)");

        this.touchCtx.fillStyle = gradient;
        const size = radius * 2;
        this.touchCtx.fillRect(touchX - radius, touchY - radius, size, size);
        this.touchCtx.globalCompositeOperation = "source-over";

        return true; // Keep active segment
      });

      // Update textures
      if (this.texture) {
        this.texture.needsUpdate = true;
      }
      this.touchTexture.needsUpdate = true;
    }

    // Note: Strokes are now built dynamically in getStrokes() from active segments
    // No need to manually clean up imageStrokes array
  }
}
