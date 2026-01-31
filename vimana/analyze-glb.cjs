const fs = require('fs');
const path = require('path');

// Simple GLB parser to extract mesh info
class SimpleGLBAnalyzer {
  constructor(glbPath) {
    this.glbPath = glbPath;
    this.logFile = path.join(__dirname, 'analysis_results.log');
    // Clear log file on init
    try {
      fs.writeFileSync(this.logFile, '');
    } catch (e) {
      console.error('Could not write to log file:', e);
    }
  }

  log(msg) {
    console.log(msg);
    try {
      fs.appendFileSync(this.logFile, msg + '\n');
    } catch (e) { }
  }

  async analyze() {
    this.log('=== MUSICTROOM.GLB MESH ANALYSIS ===\n');
    this.log(`Loading GLB from: ${this.glbPath}\n`);

    try {
      const buffer = fs.readFileSync(this.glbPath);
      const dataView = new DataView(buffer.buffer);

      // Check GLB magic bytes
      const magic = dataView.getUint32(0, true);
      const version = dataView.getUint32(4, true);
      const length = dataView.getUint32(8, true);

      if (magic !== 0x46546C67 || version !== 2) {
        throw new Error('Invalid GLB file');
      }

      this.log(`GLB Version: ${version}`);
      this.log(`Total Length: ${length} bytes\n`);

      // Find JSON chunk
      let offset = 12;
      let jsonChunk = null;

      while (offset < dataView.byteLength) {
        const chunkLength = dataView.getUint32(offset, true);
        const chunkType = this.readString(dataView, offset + 4, 4);

        if (chunkType === 'JSON') {
          const chunkData = buffer.slice(offset + 8, offset + 8 + chunkLength);
          const jsonString = chunkData.toString('utf8');
          jsonChunk = JSON.parse(jsonString);
        }

        // Skip chunk data and move to next header
        offset += 8 + chunkLength;
      }

      if (!jsonChunk) {
        throw new Error('No JSON chunk found');
      }

      // Extract mesh info from glTF structure
      this.extractMeshInfo(jsonChunk);

      // Extract light info
      this.extractLightInfo(jsonChunk);

    } catch (error) {
      console.error('Error loading GLB:', error.message);
      this.log(`Error: ${error.message}`);
    }
  }

  readString(dataView, offset, length) {
    let str = '';
    for (let i = 0; i < length; i++) {
      str += String.fromCharCode(dataView.getUint8(offset + i));
    }
    return str;
  }

  extractMeshInfo(gltf) {
    // Basic mesh info extraction (simplified for repair)
    const scene = gltf.scenes[gltf.scene || 0];
    if (!scene) {
      this.log('No scene found in glTF');
      return;
    }
    this.log('--- Geometry Analysis Skipped (Focusing on Lights) ---');
  }

  extractLightInfo(gltf) {
    this.log('\n=== LIGHT ANALYSIS ===');

    if (!gltf.extensions || !gltf.extensions.KHR_lights_punctual) {
      this.log('No KHR_lights_punctual extension found (No lights in GLB).');
      return;
    }

    const lights = gltf.extensions.KHR_lights_punctual.lights;
    this.log(`Found ${lights.length} lights:\n`);

    lights.forEach((light, index) => {
      this.log(`Light #${index}: ${light.name || '<unnamed>'}`);
      this.log(`  Type: ${light.type}`);
      this.log(`  Intensity: ${light.intensity} (This is the raw value)`);
      if (light.color) this.log(`  Color: ${light.color}`);
      if (light.range) this.log(`  Range: ${light.range}`);
      this.log('---');
    });
  }
}

// Run analysis
const glbPath = path.join(__dirname, 'public/assets/models/musicroom.glb');
const analyzer = new SimpleGLBAnalyzer(glbPath);
analyzer.analyze();
