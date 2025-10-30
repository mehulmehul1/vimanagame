import * as THREE from "three";

export class StrokeMesh {
  constructor(scene, options = {}) {
    this.scene = scene;
    this.scale = options.scale || 2;
    this.color = options.color || new THREE.Color(0xaaeeff);
    this.position = options.position || { x: 0, y: 0, z: 0 };
    this.rotation = options.rotation || { x: 0, y: 0, z: 0 };
    this.renderOrder =
      options.renderOrder !== undefined ? options.renderOrder : 10000;
    this.isStatic = options.isStatic || false;

    this.strokeGeometry = new THREE.BufferGeometry();
    this.strokeSegments = [];
    this.time = 0;
    this.strokeScaleFactor = 1.0;

    this.fadeDuration = options.fadeDuration || 6.0;
    this.rapidFadeDuration = options.rapidFadeDuration || 0.8;

    this.createMaterial();
    this.mesh = new THREE.Mesh(this.strokeGeometry, this.material);
    this.mesh.position.set(this.position.x, this.position.y, this.position.z);
    this.mesh.rotation.set(this.rotation.x, this.rotation.y, this.rotation.z);
    this.mesh.renderOrder = this.renderOrder;
    this.scene.add(this.mesh);
  }

  createMaterial() {
    const textureLoader = new THREE.TextureLoader();
    const brushTexture = textureLoader.load("/images/particle.png");
    brushTexture.minFilter = THREE.LinearFilter;
    brushTexture.magFilter = THREE.LinearFilter;
    brushTexture.wrapS = THREE.RepeatWrapping;
    brushTexture.wrapT = THREE.ClampToEdgeWrapping;

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uColor: {
          value: new THREE.Color().copy(this.color).multiplyScalar(0.8),
        },
        uBrushTexture: { value: brushTexture },
        uUseBrushTexture: { value: 1.0 },
      },
      vertexShader: `
        attribute float segmentProgress;
        attribute float segmentLength;
        attribute float segmentTime;
        attribute float fadeAlpha;
        
        uniform float uTime;
        
        varying float vProgress;
        varying float vFadeAlpha;
        varying float vSegmentLength;
        varying vec2 vUv;
        
        void main() {
          vProgress = segmentProgress;
          vFadeAlpha = fadeAlpha;
          vSegmentLength = segmentLength;
          vUv = uv;
          
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
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
        
        float hash(float n) {
          return fract(sin(n) * 43758.5453123);
        }
        
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
          float scrollSpeed = .5;
          float scrollOffset = uTime * scrollSpeed;
          vec2 brushUV = vec2(fract(vUv.x + scrollOffset), vProgress);
          vec4 texColor = texture2D(uBrushTexture, brushUV);
          
          float centerDist = abs(vProgress - 0.5) * 2.0;
          float baseCore = 1.0 - smoothstep(0.0, 1.0, centerDist);
          baseCore = pow(baseCore, 1.2);
          
          float detail = mix(0.55, 1.0, texColor.r);
          float strokeAlpha = baseCore * detail;
          
          float globalPulse = 0.5 + 0.5 * sin(uTime * 6.0);
          float alphaPulse = mix(0.8, 1.35, globalPulse);
          float brightnessPulse = mix(1.1, 2.2, globalPulse);
          
          float band = 0.85 + 0.15 * sin(vUv.x * 10.0);
          
          float alpha = strokeAlpha * vFadeAlpha * alphaPulse * band;
          alpha = clamp(alpha, 0.0, 1.2);
          
          if (alpha < 0.01) discard;
          
          vec3 finalColor = uColor * brightnessPulse;
          
          gl_FragColor = vec4(finalColor, min(alpha, 1.0));
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
      side: THREE.DoubleSide,
    });
  }

  setColor(color) {
    this.color.copy(color);
    this.material.uniforms.uColor.value.copy(color).multiplyScalar(0.8);
  }

  addStrokeSegment(segment) {
    this.strokeSegments.push({
      ...segment,
      timestamp: this.isStatic ? 0 : this.time,
      rapidFade: false,
    });
  }

  setStrokeData(strokeData) {
    this.strokeSegments = [];
    let uvPosition = 0;
    const uvScale = 8.0;
    let strokeIndex = 0;

    // Support both single stroke (array of points) and multiple strokes (array of arrays)
    const strokes =
      Array.isArray(strokeData[0]) && strokeData[0].x === undefined
        ? strokeData
        : [strokeData];

    for (const stroke of strokes) {
      uvPosition = 0;

      for (let i = 0; i < stroke.length - 1; i++) {
        const p1 = stroke[i];
        const p2 = stroke[i + 1];

        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const segmentLength = Math.sqrt(dx * dx + dy * dy);

        const uvStart = uvPosition;
        uvPosition += segmentLength * uvScale;
        const uvEnd = uvPosition;

        this.strokeSegments.push({
          uvX1: p1.x,
          uvY1: p1.y,
          uvX2: p2.x,
          uvY2: p2.y,
          uvStart,
          uvEnd,
          timestamp: 0,
          rapidFade: false,
          strokeIndex: strokeIndex,
        });
      }

      strokeIndex++;
    }

    this.updateGeometry();
  }

  updateGeometry() {
    const vertices = [];
    const uvs = [];
    const segmentProgress = [];
    const segmentLength = [];
    const segmentTime = [];
    const fadeAlpha = [];
    const indices = [];

    let vertexIndex = 0;
    const width = 0.08 * this.strokeScaleFactor;

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

        const uCoord = point.u !== undefined ? point.u : 0.0;
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

      const x1 = (segment.uvX1 - 0.5) * this.scale;
      const y1 = -(segment.uvY1 - 0.5) * this.scale;
      const x2 = (segment.uvX2 - 0.5) * this.scale;
      const y2 = -(segment.uvY2 - 0.5) * this.scale;

      const dx = x2 - x1;
      const dy = y2 - y1;
      const segmentLengthWorld = Math.sqrt(dx * dx + dy * dy);

      if (segmentLengthWorld < 0.00001) continue;

      let alpha = 1.0;

      if (!this.isStatic) {
        if (segment.rapidFade) {
          const rapidAge = this.time - segment.rapidFadeStartTime;
          if (rapidAge > this.rapidFadeDuration) {
            flushCurrentStroke();
            continue;
          }
          alpha = 1.0 - rapidAge / this.rapidFadeDuration;
        } else {
          const age = this.time - segment.timestamp;
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

    if (vertices.length > 0) {
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

      this.strokeGeometry.attributes.position.needsUpdate = true;
      this.strokeGeometry.computeBoundingSphere();
    } else {
      this.strokeGeometry.setIndex([]);
      if (this.strokeGeometry.attributes.position) {
        this.strokeGeometry.deleteAttribute("position");
      }
    }
  }

  update(dt = 0.016) {
    this.time += dt;

    if (this.material) {
      this.material.uniforms.uTime.value = this.time;
    }

    if (!this.isStatic) {
      this.strokeSegments = this.strokeSegments.filter((segment) => {
        if (segment.rapidFade) {
          const rapidAge = this.time - segment.rapidFadeStartTime;
          return rapidAge <= this.rapidFadeDuration;
        } else {
          const age = this.time - segment.timestamp;
          return age <= this.fadeDuration;
        }
      });

      if (this.strokeSegments.length > 0) {
        this.updateGeometry();
      }
    }
  }

  dispose() {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.strokeGeometry.dispose();
      this.material.dispose();
    }
  }
}
