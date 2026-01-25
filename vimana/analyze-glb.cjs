const fs = require('fs');
const path = require('path');

// Simple GLB parser to extract mesh info
class SimpleGLBAnalyzer {
  constructor(glbPath) {
    this.glbPath = glbPath;
  }

  async analyze() {
    console.log('=== MUSICTROOM.GLB MESH ANALYSIS ===\n');
    console.log(`Loading GLB from: ${this.glbPath}\n`);

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
      
      console.log(`GLB Version: ${version}`);
      console.log(`Total Length: ${length} bytes\n`);
      
      // Find JSON chunk
      let offset = 12;
      let jsonChunk = null;
      let binaryChunk = null;
      
      while (offset < dataView.byteLength) {
        const chunkLength = dataView.getUint32(offset, true);
        const chunkType = this.readString(dataView, offset + 4, 4);
        
        if (chunkType === 'JSON') {
          const chunkData = buffer.slice(offset + 8, offset + 8 + chunkLength);
          const jsonString = chunkData.toString('utf8');
          jsonChunk = JSON.parse(jsonString);
        } else if (chunkType === 'BIN') {
          binaryChunk = buffer.slice(offset + 8, offset + 8 + chunkLength);
        }
        
        offset += 8 + chunkLength;
      }
      
      if (!jsonChunk) {
        throw new Error('No JSON chunk found');
      }
      
      // Extract mesh info from glTF structure
      this.extractMeshInfo(jsonChunk);
      
    } catch (error) {
      console.error('Error loading GLB:', error.message);
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
    const meshes = [];
    
    // Get scene
    const scene = gltf.scenes[gltf.scene || 0];
    if (!scene) {
      console.log('No scene found in glTF');
      return;
    }
    
    // Get all nodes recursively
    const visitNode = (nodeIndex, depth = 0) => {
      const node = gltf.nodes[nodeIndex];
      if (!node) return;
      
      const indent = '  '.repeat(depth);
      const nodeName = node.name || '<unnamed>';
      
      // Check if node has mesh
      if (node.mesh !== undefined) {
        const mesh = gltf.meshes[node.mesh];
        const meshName = mesh.name || nodeName;
        
        // Get primitives
        const primitives = mesh.primitives || [];
        const vertexCount = primitives.reduce((sum, prim) => {
          const access = gltf.accessors[prim.attributes.POSITION];
          return sum + (access ? access.count : 0);
        }, 0);
        
        const triangleCount = primitives.reduce((sum, prim) => {
          const indices = gltf.accessors[prim.indices];
          return sum + (indices ? Math.floor(indices.count / 3) : 0);
        }, 0);
        
        console.log(`Mesh #${meshes.length + 1}: ${meshName}`);
        console.log(`  Node: ${nodeName}`);
        console.log(`  Vertices: ${vertexCount}`);
        console.log(`  Triangles: ${triangleCount}`);
        console.log(`  Primitives: ${primitives.length}`);
        
        if (primitives.length > 0) {
          const firstPrim = primitives[0];
          const material = gltf.materials ? gltf.materials[firstPrim.material] : null;
          if (material) {
            console.log(`  Material: ${material.name || '<unnamed>'}`);
            if (material.alphaMode !== undefined) {
              console.log(`  Alpha Mode: ${material.alphaMode}`);
            }
            if (material.alphaCutoff !== undefined) {
              console.log(`  Alpha Cutoff: ${material.alphaCutoff}`);
            }
            if (material.doubleSided !== undefined) {
              console.log(`  Double Sided: ${material.doubleSided}`);
            }
          }
        }
        
        console.log('---');
        meshes.push({
          name: meshName,
          nodeName,
          vertexCount,
          triangleCount
        });
      }
      
      // Visit children
      if (node.children) {
        node.children.forEach(childIndex => visitNode(childIndex, depth + 1));
      }
    };
    
    if (scene.nodes) {
      scene.nodes.forEach(nodeIndex => visitNode(nodeIndex));
    }
    
    console.log(`\n=== TOTAL MESHES: ${meshes.length} ===\n`);
    
    // Print hierarchy
    console.log('\n=== SCENE HIERARCHY ===');
    const printHierarchy = (nodeIndex, indent = '') => {
      const node = gltf.nodes[nodeIndex];
      if (!node) return;
      
      const name = node.name || '<unnamed>';
      const type = node.mesh !== undefined ? 'MeshNode' : 'Node';
      console.log(`${indent}${type}: ${name}`);
      
      if (node.children) {
        node.children.forEach(childIndex => printHierarchy(childIndex, indent + '  '));
      }
    };
    
    if (scene.nodes) {
      scene.nodes.forEach(nodeIndex => printHierarchy(nodeIndex));
    }
  }
}

// Run analysis
const glbPath = path.join(__dirname, 'public/assets/models/musicroom.glb');
const analyzer = new SimpleGLBAnalyzer(glbPath);
analyzer.analyze();
