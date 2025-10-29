const WIDTH = 500;
const HEIGHT = 500;
const STROKE_WEIGHT = 3;
const CROP_PADDING = 2;
const REPOS_PADDING = 2;

export class ImagePreprocessor {
  constructor() {
    this.offscreenCanvas = document.createElement("canvas");
    this.offscreenCanvas.width = 28;
    this.offscreenCanvas.height = 28;
    this.ctx = this.offscreenCanvas.getContext("2d");
  }

  getMinimumCoordinates(imageStrokes) {
    let min_x = Number.MAX_SAFE_INTEGER;
    let min_y = Number.MAX_SAFE_INTEGER;

    for (const stroke of imageStrokes) {
      for (let i = 0; i < stroke[0].length; i++) {
        min_x = Math.min(min_x, stroke[0][i]);
        min_y = Math.min(min_y, stroke[1][i]);
      }
    }

    return [Math.max(0, min_x), Math.max(0, min_y)];
  }

  repositionImage(imageStrokes) {
    const [min_x, min_y] = this.getMinimumCoordinates(imageStrokes);
    for (const stroke of imageStrokes) {
      for (let i = 0; i < stroke[0].length; i++) {
        stroke[0][i] = stroke[0][i] - min_x + REPOS_PADDING;
        stroke[1][i] = stroke[1][i] - min_y + REPOS_PADDING;
      }
    }
  }

  getBoundingBox(imageStrokes) {
    this.repositionImage(imageStrokes);

    const coords_x = [];
    const coords_y = [];

    for (const stroke of imageStrokes) {
      for (let i = 0; i < stroke[0].length; i++) {
        coords_x.push(stroke[0][i]);
        coords_y.push(stroke[1][i]);
      }
    }

    const x_min = Math.min(...coords_x);
    const x_max = Math.max(...coords_x);
    const y_min = Math.min(...coords_y);
    const y_max = Math.max(...coords_y);

    const width = Math.max(...coords_x) - Math.min(...coords_x);
    const height = Math.max(...coords_y) - Math.min(...coords_y);

    const coords_min = {
      x: Math.max(0, x_min - CROP_PADDING),
      y: Math.max(0, y_min - CROP_PADDING),
    };

    let coords_max;

    if (width > height) {
      coords_max = {
        x: Math.min(WIDTH, x_max + CROP_PADDING),
        y: Math.max(0, y_min + CROP_PADDING) + width,
      };
    } else {
      coords_max = {
        x: Math.max(0, x_min + CROP_PADDING) + height,
        y: Math.min(HEIGHT, y_max + CROP_PADDING),
      };
    }

    return {
      min: coords_min,
      max: coords_max,
    };
  }

  preprocessImage(imageStrokes) {
    if (!imageStrokes || imageStrokes.length === 0) {
      return null;
    }

    const strokesCopy = JSON.parse(JSON.stringify(imageStrokes));
    const { min, max } = this.getBoundingBox(strokesCopy);

    const cropWidth = max.x - min.x;
    const cropHeight = max.y - min.y;

    // Calculate scale to fill as much of 28x28 as possible while maintaining aspect ratio
    const maxDimension = Math.max(cropWidth, cropHeight);
    const targetSize = 24; // Use 24 instead of 28 to leave a small margin
    const scale = targetSize / maxDimension;

    const scaledWidth = cropWidth * scale;
    const scaledHeight = cropHeight * scale;

    // Center the drawing in 28x28 space
    const offsetX = (28 - scaledWidth) / 2;
    const offsetY = (28 - scaledHeight) / 2;

    // Draw to final 28x28 canvas directly with scaling and centering
    this.ctx.fillStyle = "white";
    this.ctx.fillRect(0, 0, 28, 28);
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = "high";

    // Match the working demo's stroke darkness (darker gray, not pure black)
    this.ctx.strokeStyle = "rgb(80, 80, 80)"; // Dark gray like the working demo
    this.ctx.lineWidth = 2.5; // Fixed width for 28x28 space
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";

    for (const stroke of strokesCopy) {
      if (stroke[0].length < 2) continue;

      this.ctx.beginPath();
      // Transform coordinates: subtract min, scale, and center
      const startX = (stroke[0][0] - min.x) * scale + offsetX;
      const startY = (stroke[1][0] - min.y) * scale + offsetY;
      this.ctx.moveTo(startX, startY);

      for (let i = 1; i < stroke[0].length; i++) {
        const x = (stroke[0][i] - min.x) * scale + offsetX;
        const y = (stroke[1][i] - min.y) * scale + offsetY;
        this.ctx.lineTo(x, y);
      }

      this.ctx.stroke();
    }

    return this.ctx.getImageData(0, 0, 28, 28);
  }

  getCanvas() {
    return this.offscreenCanvas;
  }
}
