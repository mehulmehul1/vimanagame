# PLACEHOLDER - Add your music room 3D model here

## Model Requirements:

- **Format**: GLB (glTF Binary)
- **Recommended**: Export from Blender, Maya, or similar
- **Origin**: Place at (0, 0, 0) in your 3D software
- **Scale**: Adjust in `sceneData.js` scale property if needed

## Spawn Point Configuration:

Edit `vimana/src/sceneData.js` to adjust player spawn:

```javascript
spawn: {
  position: { x: 0, y: 0.9, z: 2 },   // x,y,z world coordinates
  rotation: { x: 0, y: 180, z: 0 },    // y = degrees, 180 = facing -Z
}
```

## Adding Colliders:

Edit `vimana/src/colliderData.js` to match your room geometry:

```javascript
{
  id: 'music_room_floor',
  type: 'box',
  position: { x: 0, y: 0, z: 0 },
  size: { x: 20, y: 0.1, z: 20 },  // Adjust to room size
}
```
