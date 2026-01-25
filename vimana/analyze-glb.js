import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as THREE from 'three';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const loader = new GLTFLoader();

console.log('=== MUSICTROOM.GLB MESH ANALYSIS ===\n');

const glbPath = resolve(__dirname, 'public/assets/models/musicroom.glb');
console.log(`Loading GLB from: ${glbPath}\n`);

loader.load(glbPath, (gltf) => {
  let meshCount = 0;
  
  gltf.scene.traverse((child) => {
    if (child.isMesh) {
      meshCount++;
      console.log(`Mesh ${meshCount}: ${child.name || '<unnamed>'}`);
      console.log(`  Position: (${child.position.x.toFixed(2)}, ${child.position.y.toFixed(2)}, ${child.position.z.toFixed(2)})`);
      console.log(`  Rotation: (${child.rotation.x.toFixed(2)}, ${child.rotation.y.toFixed(2)}, ${child.rotation.z.toFixed(2)})`);
      console.log(`  Scale: (${child.scale.x.toFixed(2)}, ${child.scale.y.toFixed(2)}, ${child.scale.z.toFixed(2)})`);
      
      if (child.material) {
        console.log(`  Material: ${child.material.name || '<unnamed>'}`);
        console.log(`  Material Type: ${child.material.type}`);
        console.log(`  Color: ${child.material.color?.getHexString() || 'N/A'}`);
        console.log(`  Transparent: ${child.material.transparent}`);
        console.log(`  Opacity: ${child.material.opacity}`);
      }
      
      if (child.geometry) {
        console.log(`  Geometry: ${child.geometry.type}`);
        const vertexCount = child.geometry.attributes.position?.count || 0;
        console.log(`  Vertices: ${vertexCount}`);
        
        if (child.geometry.index) {
          console.log(`  Triangles: ${child.geometry.index.count / 3}`);
        }
      }
      
      console.log('---');
    }
  });
  
  console.log(`\n=== TOTAL MESHES: ${meshCount} ===\n`);
  
  console.log('\n=== SCENE HIERARCHY ===');
  function printHierarchy(node, indent = '') {
    const name = node.name || '<unnamed>';
    const type = node.type;
    console.log(`${indent}${type}: ${name}`);
    
    if (node.children && node.children.length > 0) {
      node.children.forEach(child => printHierarchy(child, indent + '  '));
    }
  }
  printHierarchy(gltf.scene);
  
}, undefined, (error) => {
  console.error('Error loading GLB:', error);
});
