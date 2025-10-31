import * as THREE from "three";

const CANVAS_SIZE = 500;
const STROKE_WEIGHT = 3;
const PARTICLE_GRID_DENSITY = 100; // Particles per side (100x100 = 10000 particles, ~2x original)
const PARTICLE_BASE_SIZE = 0.0225; // Base size multiplier for particles
const TOUCH_INNER_RADIUS = 2.5;
const TOUCH_OUTER_SCALE = 2.0;
const STROKE_REPULSION_DISTANCE = 0.05;
const GALAXY_SWIRL_BASE = 0.00042;
const GALAXY_SWIRL_VARIATION = 0.00045;
const GALAXY_SWIRL_NOISE = 0.35;
const DEFAULT_GALAXY_SWIRL_INTENSITY = 0.35;
const DEFAULT_GALAXY_SWIRL_RADIAL_EXPONENT = 0.15;

export class ParticleCanvas3D {
  constructor(
    scene,
    position = { x: 0, y: 2, z: -2 },
    scale = 1,
    enableParticles = true
  ) {
    this.scene = scene;
    this.enableParticles = enableParticles;
    this.strokeRepulsionDistance = STROKE_REPULSION_DISTANCE;

    // Controls overall swirl speed of particles around center
    this.galaxySwirlIntensity = DEFAULT_GALAXY_SWIRL_INTENSITY;
    this.galaxySwirlRadialExponent = DEFAULT_GALAXY_SWIRL_RADIAL_EXPONENT;

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

    // Create invisible plane (only for ML recognition, not visual display)
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.needsUpdate = true;

    const geometry = new THREE.PlaneGeometry(scale, scale);
    const material = new THREE.MeshBasicMaterial({
      visible: false, // Hidden - ML canvas only
    });

    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.position.set(position.x, position.y, position.z);
    this.mesh.userData.isDrawingCanvas = true;
    this.scene.add(this.mesh);

    // Create magical brush stroke mesh for visual display
    this.strokeScale = scale;
    this.strokeMesh = null;
    this.strokeGeometry = null;
    this.createStrokeMesh();

    // Conditionally create particle system (in front of the plane)
    this.particleSystem = null;
    this.particleOffset = 0.05;
    this.particleVelocities = null; // Store velocities on CPU for persistence
    this.particleCount = PARTICLE_GRID_DENSITY * PARTICLE_GRID_DENSITY;

    if (this.enableParticles) {
      this.particleSystem = this.createParticleSystem(scale);
      this.particleSystem.position.set(position.x, position.y, position.z);
      this.particleSystem.renderOrder = 9999; // Render on top of splats (9998)
      this.scene.add(this.particleSystem);

      // Initialize velocity storage
      this.particleVelocities = new Float32Array(this.particleCount * 3);
    }

    this.isDrawing = false;
    this.currentStroke = [[], []];
    this.lastPoint = null;
    this.time = 0;
    this.strokeUVPosition = 0; // Track absolute UV position along stroke

    // Track stroke segments with timestamps for progressive fade
    // ML strokes are dynamically built from active segments
    this.strokeSegments = []; // Array of {x1, y1, x2, y2, timestamp, rapidFade, strokeIndex}
    this.fadeDuration = 6.0; // Seconds for stroke to fade
    this.rapidFadeDuration = 0.8; // Rapid fade when clearing (800ms)
    this.incorrectGuessFadeDuration = 2.0; // Fade duration for incorrect guesses (2 seconds)
    this.currentStrokeIndex = 0; // Counter for linking segments to strokes

    // Color cycling for success feedback - advances on each correct drawing
    this.colorStage = 0; // 0=blue, 1=orange, 2=white, then cycles back to 0
    this.baseColor = new THREE.Color(0xaaeeff); // Bright whitish-blue
    this.orangeColor = new THREE.Color(0xffddaa); // Whitish orange
    this.redColor = new THREE.Color(0xffffff); // Full white (final stage)

    // Color transition state
    this.isTransitioningColor = false;
    this.colorTransitionProgress = 0;
    this.colorTransitionDuration = 1.5; // 1.5 seconds for smooth transition
    this.currentColor = new THREE.Color().copy(this.baseColor);
    this.startColor = new THREE.Color().copy(this.baseColor);
    this.targetColor = new THREE.Color().copy(this.baseColor);

    // Jitter transition state
    this.currentJitterIntensity = 0.0;
    this.startJitterIntensity = 0.0;
    this.targetJitterIntensity = 0.0;

    // Scale pulse state
    this.isPulsing = false;
    this.pulseProgress = 0;
    this.pulseDuration = 1.0; // 0.5 seconds for pulse

    // Stroke scale state
    this.strokeScaleFactor = 1.0;
    this.isStrokePulsing = false;
    this.strokePulseProgress = 0;
    this.strokePulseDuration = 1.0; // 150ms for immediate snappy response
    this.strokeScaleTarget = 2;

    // Explosion state
    this.isExploding = false;
    this.explosionProgress = 0;
    this.explosionDuration = 2.5; // Total explosion duration in seconds
    this.explosionPhase1 = 0.3; // 0-0.3: Expand to peak (0.75s)
    this.explosionPhase2 = 0.5; // 0.3-0.5: Hold with heavy jitter (0.5s)
    this.explosionPhase3 = 0.8; // 0.5-0.8: Collapse to center (0.75s)
    // 0.8-1.0: Fade out (0.5s)
    this.skipRecreateAfterExplosion = false; // Set to true for final explosion

    // Success animation state (scaled-down explosion)
    this.isSuccessAnimating = false;
    this.successAnimProgress = 0;
    this.successAnimDuration = 1.0; // Shorter than full explosion
    this.successAnimScale = 1.0; // Scale factor (0.25 for first, 0.5 for second)

    this.clearCanvas();
  }

  createStrokeMesh() {
    // Create the geometry and material for magical brush strokes
    this.strokeGeometry = new THREE.BufferGeometry();

    // Load brush texture (optional)
    const textureLoader = new THREE.TextureLoader();
    const brushTexture = textureLoader.load("/images/particle.png");
    brushTexture.minFilter = THREE.LinearFilter;
    brushTexture.magFilter = THREE.LinearFilter;
    brushTexture.wrapS = THREE.RepeatWrapping; // Repeat along stroke length
    brushTexture.wrapT = THREE.ClampToEdgeWrapping; // Clamp across width

    const strokeMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uColor: { value: new THREE.Color(0xaaeeff).multiplyScalar(0.8) }, // Darker version of particle color
        uBrushTexture: { value: brushTexture },
        uUseBrushTexture: { value: 1.0 }, // 1.0 = use texture, 0.0 = procedural
        uFractalIntensity: { value: 0.0 }, // World disintegration effect
      },
      vertexShader: `
        attribute float segmentProgress; // 0-1 along segment
        attribute float segmentLength; // Length of this segment
        attribute float segmentTime; // When this segment was drawn
        attribute float fadeAlpha; // Alpha for progressive fade
        
        uniform float uTime;
        uniform float uFractalIntensity;
        
        varying float vProgress;
        varying float vFadeAlpha;
        varying float vSegmentLength;
        varying vec2 vUv;
        
        void main() {
          vProgress = segmentProgress;
          vFadeAlpha = fadeAlpha;
          vSegmentLength = segmentLength;
          vUv = uv;
          
          vec3 pos = position;
          
          // Apply fractal warping when world is disintegrating
          if (uFractalIntensity > 0.0) {
            float warpTime = uTime * 3.0;
            float warpX = sin(warpTime + pos.y * 15.0 + segmentTime * 2.0) * cos(warpTime * 1.3 + pos.x * 12.0);
            float warpY = cos(warpTime * 1.1 + pos.x * 13.0) * sin(warpTime * 1.7 + pos.y * 11.0);
            float warpZ = sin(warpTime * 0.9 + pos.x * 10.0) * cos(warpTime * 1.5 + pos.y * 14.0);
            
            float warpScale = 0.015 * uFractalIntensity;
            pos.x += warpX * warpScale;
            pos.y += warpY * warpScale;
            pos.z += warpZ * warpScale * 0.5;
          }
          
          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        uniform float uTime;
        uniform sampler2D uBrushTexture;
        uniform float uUseBrushTexture;
        
        varying float vProgress;
        varying float vFadeAlpha;
        varying float vSegmentLength;
        varying vec2 vUv;
        
        // Simple noise function
        float hash(float n) {
          return fract(sin(n) * 43758.5453123);
        }
        
        // Improved noise function for brush texture
        float noise2d(vec2 p) {
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
          // Scroll UV along stroke length
          float scrollSpeed = -0.5;
          float scrollOffset = uTime * scrollSpeed;
          vec2 brushUV = vec2(fract(vUv.x + scrollOffset), vProgress);
          vec4 texColor = texture2D(uBrushTexture, brushUV);

          // Base stroke body so texture never drops to zero
          float centerDist = abs(vProgress - 0.5) * 2.0;
          float baseCore = 1.0 - smoothstep(0.0, 1.0, centerDist);
          baseCore = pow(baseCore, 1.2); // Softer falloff

          // Blend texture detail with base core for continuity
          float detail = mix(0.55, 1.0, texColor.r);
          float strokeAlpha = baseCore * detail;

          // Global pulse for steady, intense flashes
          float globalPulse = 0.5 + 0.5 * sin(uTime * 6.0);
          float alphaPulse = mix(0.8, 1.35, globalPulse); // Higher minimum alpha
          float brightnessPulse = mix(1.1, 2.2, globalPulse); // Keep bright peaks, raise floor

          // Subtle spatial banding to keep energy shimmering
          float band = 0.85 + 0.15 * sin(vUv.x * 10.0);

          // Apply pulses to alpha (clamped for stability)
          float alpha = strokeAlpha * vFadeAlpha * alphaPulse * band;
          alpha = clamp(alpha, 0.0, 1.2);

          if (alpha < 0.01) discard;

          // Pulse affects brightness too - magical energy effect
          vec3 finalColor = uColor * brightnessPulse;

          gl_FragColor = vec4(finalColor, min(alpha, 1.0));
        }
      `,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: THREE.NormalBlending, // Normal blending for solid strokes
      side: THREE.DoubleSide,
    });

    this.strokeMesh = new THREE.Mesh(this.strokeGeometry, strokeMaterial);
    this.strokeMesh.position.copy(this.mesh.position);
    this.strokeMesh.quaternion.copy(this.mesh.quaternion);
    this.strokeMesh.scale.copy(this.mesh.scale);
    this.strokeMesh.renderOrder = 9999; // Same as particles, depth test handles sorting
    this.scene.add(this.strokeMesh);

    console.log("[ParticleCanvas3D] Stroke mesh created:", {
      position: this.strokeMesh.position,
      rotation: this.strokeMesh.rotation,
      visible: this.strokeMesh.visible,
    });
  }

  updateStrokeMesh() {
    if (!this.strokeGeometry) return;

    // Build geometry from active stroke segments
    const vertices = [];
    const uvs = [];
    const segmentProgress = [];
    const segmentLength = [];
    const segmentTime = [];
    const fadeAlpha = [];
    const indices = [];

    let vertexIndex = 0;
    const currentTime = this.time;

    const width = 0.05 * this.strokeScaleFactor; // Slightly narrower stroke width

    const addStripGeometry = (points) => {
      if (points.length < 2) return;

      const baseIndex = vertexIndex;

      for (let i = 0; i < points.length; i++) {
        const point = points[i];
        const prev = points[i - 1] || point;
        const next = points[i + 1] || point;

        let tx = next.x - prev.x;
        let ty = next.y - prev.y;
        let tangentLength = Math.sqrt(tx * tx + ty * ty);

        if (tangentLength < 0.0001) {
          tx = 0;
          ty = 1;
          tangentLength = 1;
        }

        tx /= tangentLength;
        ty /= tangentLength;

        const nx = -ty;
        const ny = tx;

        const uCoord =
          point.u !== undefined
            ? point.u
            : point.length !== undefined
            ? point.length * 8.0
            : 0.0;
        const timeValue = point.time;
        const alphaValue = point.alpha;
        const segLengthValue = point.segmentLength || 0.0;

        const leftX = point.x + nx * width;
        const leftY = point.y + ny * width;
        const rightX = point.x - nx * width;
        const rightY = point.y - ny * width;

        vertices.push(leftX, leftY, 0.001, rightX, rightY, 0.001);
        uvs.push(uCoord, 0.0, uCoord, 1.0);
        segmentProgress.push(0.0, 1.0);
        segmentLength.push(segLengthValue, segLengthValue);
        segmentTime.push(timeValue, timeValue);
        fadeAlpha.push(alphaValue, alphaValue);

        vertexIndex += 2;
      }

      const stripSegments = points.length - 1;
      for (let i = 0; i < stripSegments; i++) {
        const a = baseIndex + i * 2;
        const b = a + 1;
        const c = a + 2;
        const d = a + 3;
        indices.push(a, b, c, b, d, c);
      }
    };

    let currentStrokeIndex = null;
    let currentPoints = [];

    const flushCurrentStroke = () => {
      if (currentPoints.length >= 2) {
        addStripGeometry(currentPoints);
      }
      currentPoints = [];
    };

    for (const segment of this.strokeSegments) {
      if (segment.strokeIndex !== currentStrokeIndex) {
        flushCurrentStroke();
        currentStrokeIndex = segment.strokeIndex;
      }

      const x1 = (segment.uvX1 - 0.5) * this.strokeScale;
      const y1 = -(segment.uvY1 - 0.5) * this.strokeScale;
      const x2 = (segment.uvX2 - 0.5) * this.strokeScale;
      const y2 = -(segment.uvY2 - 0.5) * this.strokeScale;

      const dx = x2 - x1;
      const dy = y2 - y1;
      const segmentLengthWorld = Math.sqrt(dx * dx + dy * dy);
      if (segmentLengthWorld < 0.00001) {
        continue;
      }

      let alpha = 1.0;
      if (segment.rapidFade) {
        const fadeDuration =
          segment.customFadeDuration || this.rapidFadeDuration;
        const rapidAge = currentTime - segment.rapidFadeStartTime;
        if (rapidAge > fadeDuration) {
          flushCurrentStroke();
          continue;
        }
        alpha = 1.0 - rapidAge / fadeDuration;
      } else {
        const age = currentTime - segment.timestamp;
        if (age > this.fadeDuration) {
          flushCurrentStroke();
          continue;
        }

        const fadeStart = this.fadeDuration * 0.7;
        if (age > fadeStart) {
          const fadeProgress =
            (age - fadeStart) / (this.fadeDuration - fadeStart);
          alpha = 1.0 - fadeProgress;
        }
      }

      if (alpha <= 0.0) {
        flushCurrentStroke();
        continue;
      }

      if (currentPoints.length === 0) {
        currentPoints.push({
          x: x1,
          y: y1,
          length: segment.uvStart,
          u: segment.uvStart,
          alpha,
          time: segment.timestamp,
          segmentLength: segmentLengthWorld,
        });
      } else {
        const lastPoint = currentPoints[currentPoints.length - 1];
        const distToStart = Math.sqrt(
          Math.pow(x1 - lastPoint.x, 2) + Math.pow(y1 - lastPoint.y, 2)
        );
        if (distToStart > 0.0005) {
          flushCurrentStroke();
          currentPoints.push({
            x: x1,
            y: y1,
            length: segment.uvStart,
            u: segment.uvStart,
            alpha,
            time: segment.timestamp,
            segmentLength: segmentLengthWorld,
          });
        } else {
          lastPoint.alpha = Math.max(lastPoint.alpha, alpha);
          lastPoint.time = Math.max(lastPoint.time, segment.timestamp);
          lastPoint.u =
            lastPoint.u !== undefined ? lastPoint.u : segment.uvStart;
        }
      }

      currentPoints.push({
        x: x2,
        y: y2,
        length: segment.uvEnd,
        u: segment.uvEnd,
        alpha,
        time: segment.timestamp,
        segmentLength: segmentLengthWorld,
      });
    }

    flushCurrentStroke();

    // Update geometry
    if (vertices.length > 0) {
      if (Math.random() < 0.01) {
        console.log(
          "[ParticleCanvas3D] Building stroke mesh with",
          vertices.length / 3,
          "vertices"
        );
      }

      this.strokeGeometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(vertices, 3)
      );
      this.strokeGeometry.setAttribute(
        "uv",
        new THREE.Float32BufferAttribute(uvs, 2)
      );
      this.strokeGeometry.setAttribute(
        "segmentProgress",
        new THREE.Float32BufferAttribute(segmentProgress, 1)
      );
      this.strokeGeometry.setAttribute(
        "segmentLength",
        new THREE.Float32BufferAttribute(segmentLength, 1)
      );
      this.strokeGeometry.setAttribute(
        "segmentTime",
        new THREE.Float32BufferAttribute(segmentTime, 1)
      );
      this.strokeGeometry.setAttribute(
        "fadeAlpha",
        new THREE.Float32BufferAttribute(fadeAlpha, 1)
      );
      this.strokeGeometry.setIndex(indices);

      // Mark geometry for update
      this.strokeGeometry.attributes.position.needsUpdate = true;
      this.strokeGeometry.computeBoundingSphere();

      // Ensure stroke mesh matches main mesh transform
      if (this.strokeMesh && this.mesh) {
        this.strokeMesh.position.copy(this.mesh.position);
        this.strokeMesh.quaternion.copy(this.mesh.quaternion);
        this.strokeMesh.scale.copy(this.mesh.scale);
      }
    } else {
      // Clear geometry when no segments
      this.strokeGeometry.setIndex([]);
      if (this.strokeGeometry.attributes.position) {
        this.strokeGeometry.deleteAttribute("position");
      }
    }
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
    const homePositions = new Float32Array(numParticles * 3); // Original grid positions
    const currentPositions = new Float32Array(numParticles * 3); // Current displaced positions
    const velocities = new Float32Array(numParticles * 3); // Current velocities
    const sizes = new Float32Array(numParticles);

    // Position particles in a grid with organic edge variation
    for (let i = 0; i < numParticles; i++) {
      const x = (i % PARTICLE_GRID_DENSITY) / PARTICLE_GRID_DENSITY;
      const y = Math.floor(i / PARTICLE_GRID_DENSITY) / PARTICLE_GRID_DENSITY;

      // Center the grid
      const centeredX = x - 0.5;
      const centeredY = y - 0.5;

      // Calculate distance from edges (0 at edges, 1 at center)
      const distFromEdgeX = 1.0 - Math.abs(centeredX) * 2.0;
      const distFromEdgeY = 1.0 - Math.abs(centeredY) * 2.0;
      const distFromEdge = Math.min(distFromEdgeX, distFromEdgeY);

      // Edge decay factor: 0 at center, increases toward edges
      const edgeDecay = Math.pow(1.0 - distFromEdge, 2.5);

      // Add organic variation that increases dramatically at edges
      const edgeVariationStrength = edgeDecay * 0.15;
      const noiseX =
        (Math.random() - 0.5) * 2.0 * edgeVariationStrength * scale;
      const noiseY =
        (Math.random() - 0.5) * 2.0 * edgeVariationStrength * scale;
      const noiseZ =
        (Math.random() - 0.5) * edgeVariationStrength * scale * 0.3;

      // Home position (where particle wants to return)
      const homeX = centeredX * scale + noiseX;
      const homeY = centeredY * scale + noiseY;
      const homeZ = noiseZ;

      homePositions[i * 3 + 0] = homeX;
      homePositions[i * 3 + 1] = homeY;
      homePositions[i * 3 + 2] = homeZ;

      // Start at home position
      currentPositions[i * 3 + 0] = homeX;
      currentPositions[i * 3 + 1] = homeY;
      currentPositions[i * 3 + 2] = homeZ;

      indices[i] = i;

      // Initialize velocities to zero
      velocities[i * 3 + 0] = 0;
      velocities[i * 3 + 1] = 0;
      velocities[i * 3 + 2] = 0;

      // Random size between 0.25 and 1.0, with smaller particles at edges
      const sizeVariation = 0.25 + Math.random() * 0.75;
      sizes[i] = sizeVariation * (1.0 - edgeDecay * 0.3);
    }

    geometry.setAttribute(
      "pindex",
      new THREE.InstancedBufferAttribute(indices, 1, false)
    );
    geometry.setAttribute(
      "homePosition",
      new THREE.InstancedBufferAttribute(homePositions, 3, false)
    );
    geometry.setAttribute(
      "currentPosition",
      new THREE.InstancedBufferAttribute(currentPositions, 3, false)
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
        uDeltaTime: { value: 0.016 }, // For physics calculations
        uSize: { value: PARTICLE_BASE_SIZE * scale },
        uColor: { value: new THREE.Color(0xaaeeff) },
        uPulseScale: { value: 1.0 },
        uJitterIntensity: { value: 0.0 },
        uExplosionFactor: { value: 0.0 },
        uImplosionFactor: { value: 0.0 },
      },
      vertexShader: `
        attribute float pindex;
        attribute vec3 homePosition;
        attribute vec3 currentPosition;
        attribute vec3 velocity;
        attribute float particleSize;
        
        uniform float uTime;
        uniform float uSize;
        uniform float uPulseScale;
        uniform float uJitterIntensity;
        uniform float uExplosionFactor;
        uniform float uImplosionFactor;
        
        varying vec2 vUv;
        varying float vVelocityMagnitude;
        
        void main() {
          vUv = uv;
          
          // Use currentPosition (updated on CPU with physics)
          vec3 displaced = currentPosition;
          
          // Calculate velocity magnitude for brightness
          vVelocityMagnitude = length(velocity);
          
          // Add energetic jitter that builds with each round
          if (uJitterIntensity > 0.0) {
            float jitterTime = uTime * 15.0;
            float jitterX = sin(jitterTime + pindex * 43.0) * cos(jitterTime * 1.3 + pindex * 17.0);
            float jitterY = cos(jitterTime * 1.1 + pindex * 31.0) * sin(jitterTime * 1.7 + pindex * 23.0);
            float jitterZ = sin(jitterTime * 0.9 + pindex * 19.0) * cos(jitterTime * 1.5 + pindex * 29.0);
            
            float jitterScale = 0.008 * uJitterIntensity;
            displaced.x += jitterX * jitterScale;
            displaced.y += jitterY * jitterScale;
            displaced.z += jitterZ * jitterScale * 0.5;
          }
          
          // Apply explosion force: push particles outward from center
          if (uExplosionFactor > 0.0) {
            vec3 outwardDir = normalize(displaced);
            float outwardForce = 0.8 * uExplosionFactor;
            displaced.xy += outwardDir.xy * outwardForce;
            displaced.z += 0.3 * uExplosionFactor;
          }
          
          // Apply implosion force: pull particles toward center
          if (uImplosionFactor > 0.0) {
            displaced.xyz *= (1.0 - uImplosionFactor * 0.95);
          }
          
          // Transform to world space
          vec4 mvPosition = modelViewMatrix * vec4(displaced, 1.0);
          mvPosition.xyz += position * uSize * particleSize * uPulseScale;
          
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        varying vec2 vUv;
        varying float vVelocityMagnitude;
        
        void main() {
          // Circular particle shape with soft glow
          float dist = distance(vUv, vec2(0.5));
          float alpha = 1.0 - smoothstep(0.2, 0.5, dist);
          
          // Brightness increases with velocity (GPGPU-style)
          // Base brightness is high, velocity adds extra glow
          float velocityBrightness = clamp(vVelocityMagnitude * 20.0, 0.0, 1.0);
          vec3 color = uColor * (1.5 + velocityBrightness * 1.0);
          
          // Strong base alpha with velocity boost
          float velocityAlpha = 1.0 + velocityBrightness * 0.3;
          alpha *= velocityAlpha;
          
          // Soft glow falloff
          alpha *= alpha;
          
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
          segment.customFadeDuration = null; // Use default rapid fade
        }
      }
    }

    // Clear current stroke data
    this.currentStroke = [[], []];
    this.lastPoint = null;

    // Note: strokeSegments will be removed naturally as they fade out
    // ML strokes are built dynamically from active segments via getStrokes()
  }

  fadeIncorrectGuess() {
    // Mark all existing stroke segments for incorrect guess fade (2 seconds)
    for (let segment of this.strokeSegments) {
      if (!segment.rapidFade) {
        segment.rapidFade = true;
        segment.rapidFadeStartTime = this.time;
        segment.rapidFadeStartAlpha =
          1.0 - (this.time - segment.timestamp) / this.fadeDuration;
        segment.customFadeDuration = this.incorrectGuessFadeDuration; // Use 2 second fade
      }
    }
  }

  startStroke(uv) {
    if (!uv) return;

    this.isDrawing = true;
    this.strokeUVPosition = 0; // Reset UV position for new stroke

    // Record on high-res canvas for ML
    const x = Math.floor(uv.x * CANVAS_SIZE);
    const y = Math.floor((1 - uv.y) * CANVAS_SIZE);

    this.lastPoint = { x, y, uvX: uv.x, uvY: 1 - uv.y };
    this.currentStroke = [[x], [y]];
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

    // Calculate segment length in world space for UV mapping
    const worldX1 = this.lastPoint.uvX - 0.5;
    const worldY1 = -(this.lastPoint.uvY - 0.5);
    const worldX2 = uv.x - 0.5;
    const worldY2 = -(1 - uv.y - 0.5);
    const segmentLength = Math.sqrt(
      Math.pow(worldX2 - worldX1, 2) + Math.pow(worldY2 - worldY1, 2)
    );

    // Store absolute UV position for this segment
    // Scale the length to control texture repeat frequency (higher = more repeats)
    const uvScale = 8.0; // Higher value = denser texture tiling along visual stroke
    const uvStart = this.strokeUVPosition;
    this.strokeUVPosition += segmentLength * uvScale;
    const uvEnd = this.strokeUVPosition;

    // Record stroke segment with timestamp for progressive fade
    this.strokeSegments.push({
      x1: this.lastPoint.x,
      y1: this.lastPoint.y,
      x2: x,
      y2: y,
      uvX1: this.lastPoint.uvX || uv.x,
      uvY1: this.lastPoint.uvY || 1 - uv.y,
      uvX2: uv.x,
      uvY2: 1 - uv.y,
      uvStart: uvStart, // Absolute UV position at segment start
      uvEnd: uvEnd, // Absolute UV position at segment end
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

    this.lastPoint = { x, y, uvX: uv.x, uvY: 1 - uv.y };
  }

  stampTouchTexture(touchX, touchY, intensity = 1.0) {
    const innerRadius = TOUCH_INNER_RADIUS;
    const outerRadius = innerRadius * TOUCH_OUTER_SCALE;
    const clamped = Math.min(Math.max(intensity, 0), 1);

    this.touchCtx.globalCompositeOperation = "lighter";

    const innerGradient = this.touchCtx.createRadialGradient(
      touchX,
      touchY,
      0,
      touchX,
      touchY,
      innerRadius
    );
    innerGradient.addColorStop(0.0, `rgba(255, 255, 255, ${clamped})`);
    innerGradient.addColorStop(0.5, `rgba(255, 255, 255, ${clamped * 0.9})`);
    innerGradient.addColorStop(1.0, "rgba(255, 255, 255, 0)");

    this.touchCtx.fillStyle = innerGradient;
    const innerSize = innerRadius * 2;
    this.touchCtx.fillRect(
      touchX - innerRadius,
      touchY - innerRadius,
      innerSize,
      innerSize
    );

    const outerGradient = this.touchCtx.createRadialGradient(
      touchX,
      touchY,
      innerRadius * 0.6,
      touchX,
      touchY,
      outerRadius
    );
    const outerAlpha = clamped * 0.35;
    outerGradient.addColorStop(0.0, `rgba(255, 255, 255, ${outerAlpha})`);
    outerGradient.addColorStop(0.7, `rgba(255, 255, 255, ${outerAlpha * 0.6})`);
    outerGradient.addColorStop(1.0, "rgba(255, 255, 255, 0)");

    this.touchCtx.fillStyle = outerGradient;
    const outerSize = outerRadius * 2;
    this.touchCtx.fillRect(
      touchX - outerRadius,
      touchY - outerRadius,
      outerSize,
      outerSize
    );

    this.touchCtx.globalCompositeOperation = "source-over";
  }

  drawToTouchTexture(x, y) {
    const touchX = x * PARTICLE_GRID_DENSITY;
    const touchY = y * PARTICLE_GRID_DENSITY;

    this.stampTouchTexture(touchX, touchY, 1.0);

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
        const fadeDuration =
          segment.customFadeDuration || this.rapidFadeDuration;
        const rapidAge = this.time - segment.rapidFadeStartTime;
        isExpired = rapidAge > fadeDuration;
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
        const fadeDuration =
          segment.customFadeDuration || this.rapidFadeDuration;
        const rapidAge = currentTime - segment.rapidFadeStartTime;
        isExpired = rapidAge > fadeDuration;
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

  triggerColorCycle() {
    // Check if we're about to transition from red (2) back to blue (0)
    const nextStage = (this.colorStage + 1) % 3;

    if (this.colorStage === 2 && nextStage === 0) {
      // Red -> Blue: trigger explosion animation instead
      console.log(
        "[ParticleCanvas3D] triggerColorCycle detected red->blue, triggering explosion"
      );
      this.triggerExplosion();
      return;
    }

    // Advance to next color stage on each success
    console.log(
      `[ParticleCanvas3D] triggerColorCycle: advancing from stage ${this.colorStage} to ${nextStage}`
    );
    this.colorStage = nextStage;

    if (!this.particleSystem || !this.particleSystem.material) return;

    // Determine target color and jitter intensity based on stage
    let newTargetColor;
    let jitterIntensity;
    switch (this.colorStage) {
      case 0:
        newTargetColor = this.baseColor; // Blue
        jitterIntensity = 0.0; // Calm
        break;
      case 1:
        newTargetColor = this.orangeColor; // Orange
        jitterIntensity = 0.15; // Medium energy
        break;
      case 2:
        newTargetColor = this.redColor; // White (final stage)
        jitterIntensity = 0.5; // Intense, about to burst
        break;
    }

    // Start transition from current color to new target color
    this.startColor.copy(this.currentColor);
    this.targetColor.copy(newTargetColor);
    this.startJitterIntensity = this.currentJitterIntensity;
    this.targetJitterIntensity = jitterIntensity;
    this.isTransitioningColor = true;
    this.colorTransitionProgress = 0;
  }

  triggerExplosion() {
    this.isExploding = true;
    this.explosionProgress = 0;
    console.log(
      `[ParticleCanvas3D] Explosion triggered! isExploding=${this.isExploding}`
    );
  }

  triggerSuccessAnimation(scale = 0.25) {
    // Scale factor: 0.25 for first success, 0.5 for second success
    this.isSuccessAnimating = true;
    this.successAnimProgress = 0;
    this.successAnimScale = scale;
    console.log(
      `[ParticleCanvas3D] Success animation triggered! scale=${scale}`
    );
  }

  triggerPulse() {
    // Start pulse animation
    this.isPulsing = true;
    this.pulseProgress = 0;
  }

  triggerStrokePulse() {
    this.isStrokePulsing = true;
    this.strokePulseProgress = 0;
  }

  triggerExplosion(explosionCenter, explosionRadius, explosionForce) {
    if (!this.particleSystem || !this.particleSystem.material) return;

    const explosionFactor = explosionForce * 0.005; // Scale explosion force
    this.particleSystem.material.uniforms.uExplosionFactor.value =
      explosionFactor;

    // Reset explosion factor after a short duration
    setTimeout(() => {
      this.particleSystem.material.uniforms.uExplosionFactor.value = 0.0;
    }, 100); // 100ms duration for explosion
  }

  setBrushTexture(enabled) {
    // Toggle brush texture on/off
    if (this.strokeMesh && this.strokeMesh.material) {
      this.strokeMesh.material.uniforms.uUseBrushTexture.value = enabled
        ? 1.0
        : 0.0;
      console.log(
        "[ParticleCanvas3D] Brush texture",
        enabled ? "enabled" : "disabled"
      );
    }
  }

  updateParticlePhysics(dt) {
    if (!this.particleSystem || !this.particleVelocities) return;

    const geometry = this.particleSystem.geometry;
    const homePositions = geometry.attributes.homePosition.array;
    const currentPositions = geometry.attributes.currentPosition.array;
    const velocities = geometry.attributes.velocity.array;

    // Physics constants (GPGPU-inspired)
    const VELOCITY_RELAXATION = 0.9; // Velocity damping (reduced for more float)
    const HOME_ATTRACTION_STRENGTH = 0.0003; // Pull back to home position (reduced to allow more drift)
    const STROKE_REPULSION_STRENGTH = 0.003; // Push away from strokes
    const MAX_REPULSION_DISTANCE = this.strokeRepulsionDistance; // Max distance for stroke influence
    const AMBIENT_DRIFT_STRENGTH = 0.00008; // Gentle floating motion

    for (let i = 0; i < this.particleCount; i++) {
      const i3 = i * 3;

      // Current particle data
      const px = currentPositions[i3];
      const py = currentPositions[i3 + 1];
      const pz = currentPositions[i3 + 2];

      const hx = homePositions[i3];
      const hy = homePositions[i3 + 1];
      const hz = homePositions[i3 + 2];

      let vx = velocities[i3];
      let vy = velocities[i3 + 1];
      let vz = velocities[i3 + 2];

      // Apply velocity relaxation (particles gradually slow down)
      vx *= VELOCITY_RELAXATION;
      vy *= VELOCITY_RELAXATION;
      vz *= VELOCITY_RELAXATION;

      // Force 1: Ambient drift (gentle floating motion like clouds)
      const timeOffset = this.time + i * 0.1;
      const driftX =
        Math.sin(timeOffset * 0.3 + i * 0.05) * Math.cos(timeOffset * 0.17);
      const driftY =
        Math.cos(timeOffset * 0.25 + i * 0.07) * Math.sin(timeOffset * 0.19);
      const driftZ =
        Math.sin(timeOffset * 0.15 + i * 0.03) * Math.cos(timeOffset * 0.21);

      vx += driftX * AMBIENT_DRIFT_STRENGTH;
      vy += driftY * AMBIENT_DRIFT_STRENGTH;
      vz += driftZ * AMBIENT_DRIFT_STRENGTH * 0.5;

      // Force 2: Galaxy-style swirl around center
      const radius = Math.sqrt(px * px + py * py);
      const effectiveRadius = this.strokeScale * 0.9 + 0.0001;
      const radiusFactor = Math.min(radius / effectiveRadius, 1.0);

      if (radius > 0.0005) {
        const swirlDirX = -py / radius;
        const swirlDirY = px / radius;

        const radialInfluence = Math.pow(
          radiusFactor,
          this.galaxySwirlRadialExponent
        );

        const swirlStrength =
          (GALAXY_SWIRL_BASE + GALAXY_SWIRL_VARIATION * radialInfluence) *
          radialInfluence;

        const swirlNoise =
          1.0 + Math.sin(this.time * 0.9 + i * 0.13) * GALAXY_SWIRL_NOISE;

        const swirlIntensity = this.galaxySwirlIntensity;

        vx += swirlDirX * swirlStrength * swirlNoise * swirlIntensity;
        vy += swirlDirY * swirlStrength * swirlNoise * swirlIntensity;
      }

      // Force 3: Attraction to home position (but loosely, to maintain organic spread)
      const dx = hx - px;
      const dy = hy - py;
      const dz = hz - pz;
      const distToHome = Math.sqrt(dx * dx + dy * dy + dz * dz);

      if (distToHome > 0.001) {
        const dirX = dx / distToHome;
        const dirY = dy / distToHome;
        const dirZ = dz / distToHome;

        // Only pull back if particle is getting too far from home
        const pullStrength =
          distToHome > 0.02
            ? HOME_ATTRACTION_STRENGTH
            : HOME_ATTRACTION_STRENGTH * 0.1;

        vx += dirX * pullStrength;
        vy += dirY * pullStrength;
        vz += dirZ * pullStrength;
      }

      // Force 4: Repulsion from stroke segments
      for (const segment of this.strokeSegments) {
        const worldX = (segment.uvX2 - 0.5) * this.strokeScale;
        const worldY = -(segment.uvY2 - 0.5) * this.strokeScale;

        const sdx = px - worldX;
        const sdy = py - worldY;
        const distToStroke = Math.sqrt(sdx * sdx + sdy * sdy);

        if (distToStroke < MAX_REPULSION_DISTANCE && distToStroke > 0.001) {
          const pushStrength =
            (1.0 - distToStroke / MAX_REPULSION_DISTANCE) *
            STROKE_REPULSION_STRENGTH;

          const pushDirX = sdx / distToStroke;
          const pushDirY = sdy / distToStroke;

          vx += pushDirX * pushStrength;
          vy += pushDirY * pushStrength;
        }
      }

      // Update velocity
      velocities[i3] = vx;
      velocities[i3 + 1] = vy;
      velocities[i3 + 2] = vz;

      // Update position based on velocity
      currentPositions[i3] = px + vx;
      currentPositions[i3 + 1] = py + vy;
      currentPositions[i3 + 2] = pz + vz;
    }

    // Mark attributes for update
    geometry.attributes.velocity.needsUpdate = true;
    geometry.attributes.currentPosition.needsUpdate = true;
  }

  getMesh() {
    return this.mesh;
  }

  dispose() {
    // Dispose invisible plane
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh.material.dispose();
      if (this.texture) {
        this.texture.dispose();
      }
    }

    // Dispose stroke mesh
    if (this.strokeMesh) {
      this.scene.remove(this.strokeMesh);
      if (this.strokeGeometry) {
        this.strokeGeometry.dispose();
      }
      this.strokeMesh.material.dispose();
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

    // Update particle physics (GPGPU-inspired velocity-based motion)
    if (!this.isExploding && !this.isSuccessAnimating) {
      this.updateParticlePhysics(dt);
    }

    // Update explosion animation progress (happens regardless of particle system)
    if (this.isExploding) {
      this.explosionProgress += dt / this.explosionDuration;

      if (Math.random() < 0.02) {
        // Log occasionally to avoid spam
        console.log(
          `[ParticleCanvas3D] Explosion progress: ${(
            this.explosionProgress * 100
          ).toFixed(1)}%`
        );
      }

      if (this.explosionProgress >= 1.0) {
        // Explosion complete
        console.log("[ParticleCanvas3D] Explosion complete");
        this.isExploding = false;
        this.explosionProgress = 0;

        // Dispose old particle system
        if (this.particleSystem) {
          this.scene.remove(this.particleSystem);
          this.particleSystem.geometry.dispose();
          this.particleSystem.material.dispose();
          this.particleSystem = null;
        }

        // Only recreate if this isn't the final explosion
        if (!this.skipRecreateAfterExplosion) {
          console.log(
            "[ParticleCanvas3D] Recreating particle system for next round"
          );

          // Create fresh particle system
          if (this.enableParticles) {
            this.particleSystem = this.createParticleSystem(this.strokeScale);
            this.particleSystem.position.copy(this.mesh.position);
            this.particleSystem.renderOrder = 9999;
            this.scene.add(this.particleSystem);

            // Reinitialize velocity storage
            this.particleVelocities = new Float32Array(this.particleCount * 3);
          }

          // Reset to blue stage for next round
          this.colorStage = 0;
          this.currentColor.copy(this.baseColor);
          this.currentJitterIntensity = 0.0;

          if (this.particleSystem && this.particleSystem.material) {
            this.particleSystem.material.uniforms.uColor.value.copy(
              this.currentColor
            );
            this.particleSystem.material.uniforms.uJitterIntensity.value = 0.0;
            this.particleSystem.material.uniforms.uPulseScale.value = 1.0;
          }

          // Clear strokes
          this.clearCanvas();
        } else {
          console.log(
            "[ParticleCanvas3D] Skipping recreation - final explosion"
          );
        }
      }
    }

    // Update success animation progress
    if (this.isSuccessAnimating) {
      this.successAnimProgress += dt / this.successAnimDuration;

      if (this.successAnimProgress >= 1.0) {
        // Success animation complete
        this.isSuccessAnimating = false;
        this.successAnimProgress = 0;

        // Reset uniforms back to normal
        if (this.particleSystem && this.particleSystem.material) {
          this.particleSystem.material.uniforms.uPulseScale.value = 1.0;
          this.particleSystem.material.uniforms.uJitterIntensity.value =
            this.currentJitterIntensity || 0;
          this.particleSystem.material.uniforms.uExplosionFactor.value = 0;
          this.particleSystem.material.uniforms.uImplosionFactor.value = 0;
        }
      }
    }

    // Update shader time
    if (this.particleSystem && this.particleSystem.material) {
      this.particleSystem.material.uniforms.uTime.value = this.time;

      // Animate explosion effects if currently exploding
      if (this.isExploding && this.explosionProgress < 1.0) {
        const t = this.explosionProgress;

        if (t < this.explosionPhase1) {
          // Phase 1 (0-0.3): Expand to peak
          const phase1Progress = t / this.explosionPhase1; // 0 -> 1

          // Scale up particles
          const pulseScale = 1.0 + phase1Progress * 1.2; // 1.0 -> 2.2
          this.particleSystem.material.uniforms.uPulseScale.value = pulseScale;

          // Build jitter
          const jitter = 0.5 + phase1Progress * 1.0; // 0.5 -> 1.5
          this.particleSystem.material.uniforms.uJitterIntensity.value = jitter;

          // Push outward
          const explosionFactor = phase1Progress * 0.6; // 0 -> 0.6
          this.particleSystem.material.uniforms.uExplosionFactor.value =
            explosionFactor;
          this.particleSystem.material.uniforms.uImplosionFactor.value = 0;
        } else if (t < this.explosionPhase2) {
          // Phase 2 (0.3-0.5): Hold at peak with HEAVY jitter

          // Keep large
          this.particleSystem.material.uniforms.uPulseScale.value = 2.2;

          // INTENSE jitter
          this.particleSystem.material.uniforms.uJitterIntensity.value = 2.5;

          // Hold expansion
          this.particleSystem.material.uniforms.uExplosionFactor.value = 0.6;
          this.particleSystem.material.uniforms.uImplosionFactor.value = 0;
        } else if (t < this.explosionPhase3) {
          // Phase 3 (0.5-0.8): Collapse to center
          const phase3Progress =
            (t - this.explosionPhase2) /
            (this.explosionPhase3 - this.explosionPhase2); // 0 -> 1

          // Shrink particles as they collapse
          const pulseScale = 2.2 - phase3Progress * 2.1; // 2.2 -> 0.1
          this.particleSystem.material.uniforms.uPulseScale.value = pulseScale;

          // Jitter decreases during collapse
          const jitter = 2.5 * (1.0 - phase3Progress); // 2.5 -> 0
          this.particleSystem.material.uniforms.uJitterIntensity.value = jitter;

          // Smoothly transition explosion force out while implosion force in
          const explosionFactor = 0.6 * (1.0 - phase3Progress); // 0.6 -> 0
          this.particleSystem.material.uniforms.uExplosionFactor.value =
            explosionFactor;
          this.particleSystem.material.uniforms.uImplosionFactor.value =
            phase3Progress; // 0 -> 1
        } else {
          // Phase 4 (0.8-1.0): Fade out at singularity
          const phase4Progress =
            (t - this.explosionPhase3) / (1.0 - this.explosionPhase3); // 0 -> 1

          // Shrink to nothing
          const pulseScale = 0.1 * (1.0 - phase4Progress); // 0.1 -> 0
          this.particleSystem.material.uniforms.uPulseScale.value = pulseScale;

          // No jitter
          this.particleSystem.material.uniforms.uJitterIntensity.value = 0;

          // Full implosion
          this.particleSystem.material.uniforms.uExplosionFactor.value = 0;
          this.particleSystem.material.uniforms.uImplosionFactor.value = 1.0;
        }
      }

      // Animate success effects (scaled-down explosion)
      if (this.isSuccessAnimating && this.successAnimProgress < 1.0) {
        const t = this.successAnimProgress;
        const scale = this.successAnimScale;

        // Second success (0.5) gets higher multipliers for jitter and scale
        const isSecondSuccess = scale >= 0.45; // 0.5 scale = second success
        const scaleMultiplier = isSecondSuccess ? 1.4 : 1.0; // 40% more scale for second
        const jitterMultiplier = isSecondSuccess ? 1.8 : 1.0; // 80% more jitter for second

        // Simplified: expand to 50%, then collapse back
        if (t < 0.5) {
          // Phase 1 (0-0.5): Expand to peak
          const phase1Progress = t / 0.5; // 0 -> 1

          // Scale up particles (scaled, with multiplier for second success)
          const pulseScale =
            1.0 + phase1Progress * 1.2 * scale * scaleMultiplier;
          this.particleSystem.material.uniforms.uPulseScale.value = pulseScale;

          // Build jitter (scaled, with multiplier for second success)
          const jitter = phase1Progress * 1.5 * scale * jitterMultiplier;
          this.particleSystem.material.uniforms.uJitterIntensity.value = jitter;

          // Push outward (scaled)
          const explosionFactor = phase1Progress * 0.6 * scale; // 0 -> 0.6 * scale
          this.particleSystem.material.uniforms.uExplosionFactor.value =
            explosionFactor;
          this.particleSystem.material.uniforms.uImplosionFactor.value = 0;
        } else {
          // Phase 2 (0.5-1.0): Collapse back
          const phase2Progress = (t - 0.5) / 0.5; // 0 -> 1
          const peakScale = 1.0 + 1.2 * scale * scaleMultiplier;
          const peakJitter = 1.5 * scale * jitterMultiplier;
          const peakExplosion = 0.6 * scale;

          // Shrink particles back
          const pulseScale = peakScale - phase2Progress * (peakScale - 1.0);
          this.particleSystem.material.uniforms.uPulseScale.value = pulseScale;

          // Jitter decreases
          const jitter = peakJitter * (1.0 - phase2Progress);
          this.particleSystem.material.uniforms.uJitterIntensity.value = jitter;

          // Explosion factor decreases
          const explosionFactor = peakExplosion * (1.0 - phase2Progress);
          this.particleSystem.material.uniforms.uExplosionFactor.value =
            explosionFactor;
          this.particleSystem.material.uniforms.uImplosionFactor.value = 0;
        }
      }

      // Update color transition (and jitter ramp-up)
      if (this.isTransitioningColor) {
        this.colorTransitionProgress += dt / this.colorTransitionDuration;

        if (this.colorTransitionProgress >= 1.0) {
          // Transition complete
          this.isTransitioningColor = false;
          this.colorTransitionProgress = 0;
          this.currentColor.copy(this.targetColor);
          this.currentJitterIntensity = this.targetJitterIntensity;
        } else {
          // Smooth lerp between start and target
          const t = this.colorTransitionProgress;
          // Apply easing for smoother feel (ease-in-out)
          const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
          this.currentColor.lerpColors(
            this.startColor,
            this.targetColor,
            eased
          );

          // Lerp jitter intensity at the same time
          this.currentJitterIntensity =
            this.startJitterIntensity +
            (this.targetJitterIntensity - this.startJitterIntensity) * eased;
        }

        this.particleSystem.material.uniforms.uColor.value.copy(
          this.currentColor
        );
        this.particleSystem.material.uniforms.uJitterIntensity.value =
          this.currentJitterIntensity;
      }

      // Always update stroke color to match particle color with offset
      if (this.strokeMesh && this.strokeMesh.material) {
        const strokeColor = this.currentColor.clone();
        // Make it slightly darker and more saturated
        strokeColor.multiplyScalar(0.8);
        this.strokeMesh.material.uniforms.uColor.value.copy(strokeColor);
      }

      // Update scale pulse
      if (this.isPulsing) {
        this.pulseProgress += dt / this.pulseDuration;

        if (this.pulseProgress >= 1.0) {
          // Pulse complete
          this.isPulsing = false;
          this.pulseProgress = 0;
          this.particleSystem.material.uniforms.uPulseScale.value = 1.0;
        } else {
          // Pulse in and out: 0->1 scales up, 1->0 scales back down
          const t = this.pulseProgress;
          // Create a pulse that goes 1.0 -> 1.5 -> 1.0 (sine wave)
          const pulseScale = 1.0 + Math.sin(t * Math.PI) * 0.5;
          this.particleSystem.material.uniforms.uPulseScale.value = pulseScale;
        }
      }
    }

    // Update stroke scale pulse (on successful match)
    if (this.isStrokePulsing) {
      this.strokePulseProgress += dt / this.strokePulseDuration;

      if (this.strokePulseProgress >= 1.0) {
        // Pulse complete
        this.isStrokePulsing = false;
        this.strokePulseProgress = 0;
        this.strokeScaleFactor = 1.0;
      } else {
        // Pulse in and out: 1.0 -> 1.1 -> 1.0
        const t = this.strokePulseProgress;
        const pulseScale =
          1.0 + Math.sin(t * Math.PI) * (this.strokeScaleTarget - 1.0);
        this.strokeScaleFactor = pulseScale;
      }
    }

    // During explosion, scale stroke up at apex then shrink to zero
    if (this.isExploding && this.explosionProgress < 1.0) {
      const t = this.explosionProgress;

      if (t < this.explosionPhase2) {
        // Phases 1-2: Scale up to 10% at apex
        const phase1And2Progress = t / this.explosionPhase2;
        this.strokeScaleFactor =
          1.0 + phase1And2Progress * (this.strokeScaleTarget - 1.0);
      } else if (t < this.explosionPhase3) {
        // Phase 3: Start at 10%, shrink to zero
        const phase3Progress =
          (t - this.explosionPhase2) /
          (this.explosionPhase3 - this.explosionPhase2);
        this.strokeScaleFactor =
          this.strokeScaleTarget * (1.0 - phase3Progress);
      } else {
        // Phase 4: Already at zero or below
        this.strokeScaleFactor = 0.0;
      }
    }

    // Filter out expired segments
    if (this.strokeSegments.length > 0) {
      this.strokeSegments = this.strokeSegments.filter((segment) => {
        // Check if segment is expired
        if (segment.rapidFade) {
          const fadeDuration =
            segment.customFadeDuration || this.rapidFadeDuration;
          const rapidAge = this.time - segment.rapidFadeStartTime;
          return rapidAge <= fadeDuration;
        } else {
          const age = this.time - segment.timestamp;
          return age <= this.fadeDuration;
        }
      });

      // Rebuild stroke mesh with current segments
      this.updateStrokeMesh();
    }

    // Update stroke shader time
    if (this.strokeMesh && this.strokeMesh.material) {
      this.strokeMesh.material.uniforms.uTime.value = this.time;
    }

    // Note: Strokes are now built dynamically in getStrokes() from active segments
    // No need to manually clean up imageStrokes array
  }
}
