const WIDTH = 500;
const HEIGHT = 500;
const CROP_PADDING = 2;
const REPOS_PADDING = 2;

function isMacOS() {
  const userAgent = navigator.userAgent.toLowerCase();
  const platform = navigator.platform;
  // Detect macOS (but not iOS/iPad)
  return (
    (platform === "MacIntel" && navigator.maxTouchPoints <= 1) ||
    userAgent.includes("mac os x") ||
    userAgent.includes("macintosh")
  );
}

export class ImagePreprocessor {
  constructor() {
    // No longer need persistent canvas - created per preprocessing call
    this.isMacOS = isMacOS();
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

  async preprocessImage(imageStrokes) {
    if (!imageStrokes || imageStrokes.length === 0) {
      return null;
    }

    const strokesCopy = JSON.parse(JSON.stringify(imageStrokes));
    const { min, max } = this.getBoundingBox(strokesCopy);

    const cropWidth = max.x - min.x;
    const cropHeight = max.y - min.y;

    // Draw to a temporary larger canvas (better quality before resize)
    const tempCanvas = document.createElement("canvas");
    const tempSize = 280; // 10x target size for better quality
    tempCanvas.width = tempSize;
    tempCanvas.height = tempSize;
    const tempCtx = tempCanvas.getContext("2d", {
      // macOS-specific: use different context settings
      alpha: false,
      willReadFrequently: false,
    });

    // Calculate scale to fit in tempCanvas
    const maxDimension = Math.max(cropWidth, cropHeight);
    const scale = (tempSize * 0.85) / maxDimension; // 85% to leave margin

    const scaledWidth = cropWidth * scale;
    const scaledHeight = cropHeight * scale;
    const offsetX = (tempSize - scaledWidth) / 2;
    const offsetY = (tempSize - scaledHeight) / 2;

    // White background (QuickDraw model expects white background)
    tempCtx.fillStyle = "white";
    tempCtx.fillRect(0, 0, tempSize, tempSize);

    // Black stroke (QuickDraw model expects black strokes on white)
    tempCtx.strokeStyle = "black";

    // Use thin strokes for all platforms (model needs very light strokes)
    tempCtx.lineWidth = 3;
    tempCtx.lineCap = "round";
    tempCtx.lineJoin = "round";

    // Draw strokes
    for (const stroke of strokesCopy) {
      if (stroke[0].length < 2) continue;
      tempCtx.beginPath();
      const startX = (stroke[0][0] - min.x) * scale + offsetX;
      const startY = (stroke[1][0] - min.y) * scale + offsetY;
      tempCtx.moveTo(startX, startY);
      for (let i = 1; i < stroke[0].length; i++) {
        const x = (stroke[0][i] - min.x) * scale + offsetX;
        const y = (stroke[1][i] - min.y) * scale + offsetY;
        tempCtx.lineTo(x, y);
      }
      tempCtx.stroke();
    }

    // Resize to 28x28
    const finalCanvas = document.createElement("canvas");
    finalCanvas.width = 28;
    finalCanvas.height = 28;
    const finalCtx = finalCanvas.getContext("2d", {
      alpha: false,
      desynchronized: false,
      willReadFrequently: false,
    });

    if (this.isMacOS) {
      // macOS: Use manual pixel sampling for consistent results
      // This avoids browser-specific drawImage rendering differences
      const sourceData = tempCtx.getImageData(0, 0, tempSize, tempSize);
      const targetData = finalCtx.createImageData(28, 28);

      const scaleX = tempSize / 28;
      const scaleY = tempSize / 28;

      for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
          // Average pixels in the source area (box filter) to reduce artifacts
          const startX = Math.floor(x * scaleX);
          const startY = Math.floor(y * scaleY);
          const endX = Math.floor((x + 1) * scaleX);
          const endY = Math.floor((y + 1) * scaleY);

          let sum = 0;
          let count = 0;

          for (let sy = startY; sy < endY && sy < tempSize; sy++) {
            for (let sx = startX; sx < endX && sx < tempSize; sx++) {
              const idx = (sy * tempSize + sx) * 4;
              sum += sourceData.data[idx]; // Use red channel (grayscale)
              count++;
            }
          }

          const avg = count > 0 ? Math.round(sum / count) : 255;
          const targetIdx = (y * 28 + x) * 4;
          targetData.data[targetIdx] = avg; // R
          targetData.data[targetIdx + 1] = avg; // G
          targetData.data[targetIdx + 2] = avg; // B
          targetData.data[targetIdx + 3] = 255; // A
        }
      }

      finalCtx.putImageData(targetData, 0, 0);
    } else {
      // Windows/Other: Use standard resize
      finalCtx.imageSmoothingEnabled = true;
      finalCtx.imageSmoothingQuality = "high";
      finalCtx.drawImage(tempCanvas, 0, 0, 28, 28);
    }

    // Return the 28x28 canvas
    return finalCanvas;
  }
}
