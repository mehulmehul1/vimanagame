/**
 * createCubemap.ts - Procedural Environment Cubemap
 * ==================================================
 *
 * Creates a simple gradient cubemap for water reflections.
 * This is a basic implementation - can be replaced with loaded textures.
 *
 * Based on environment cubemap needs for WaterBall fluid rendering.
 */

export interface CubemapOptions {
    device: GPUDevice;
    size?: number;
    format?: GPUTextureFormat;
    topColor?: [number, number, number, number];
    bottomColor?: [number, number, number, number];
}

/**
 * Create a procedural gradient cubemap for reflections
 * The cubemap has a gradient from top (sky color) to bottom (horizon/fog)
 */
export function createProceduralCubemap(options: CubemapOptions): GPUTexture {
    const {
        device,
        size = 64,
        format = 'rgba8unorm',
        topColor = [0.3, 0.8, 1.0, 1.0], // Light blue sky
        bottomColor = [0.7, 0.7, 0.75, 1.0], // Fog color
    } = options;

    // Create cubemap texture
    const cubemap = device.createTexture({
        label: 'Environment Cubemap',
        size: [size, size, 6],
        format: format,
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        dimension: '2d-array',
    });

    // Generate gradient data for each face
    const faceData = new Uint8Array(size * size * 4);

    for (let face = 0; face < 6; face++) {
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const idx = (y * size + x) * 4;

                // Calculate gradient based on Y position
                // For most faces, Y corresponds to vertical direction
                let t = y / (size - 1);

                // Adjust gradient direction based on face
                // +Y (top face, face 2) should be mostly sky
                // -Y (bottom face, face 3) should be mostly ground/fog
                if (face === 2) {
                    // +Y (top)
                    t = 0.2; // Mostly sky
                } else if (face === 3) {
                    // -Y (bottom)
                    t = 0.8; // Mostly fog
                }

                // Interpolate between top and bottom colors
                const r = topColor[0] * (1 - t) + bottomColor[0] * t;
                const g = topColor[1] * (1 - t) + bottomColor[1] * t;
                const b = topColor[2] * (1 - t) + bottomColor[2] * t;
                const a = topColor[3] * (1 - t) + bottomColor[3] * t;

                faceData[idx] = Math.floor(r * 255);
                faceData[idx + 1] = Math.floor(g * 255);
                faceData[idx + 2] = Math.floor(b * 255);
                faceData[idx + 3] = Math.floor(a * 255);
            }
        }

        // Upload face data to texture
        device.queue.writeTexture(
            {
                texture: cubemap,
                origin: { x: 0, y: 0, z: face },
                mipLevel: 0,
            },
            faceData,
            { bytesPerRow: size * 4, rowsPerImage: size },
            [size, size, 1]
        );
    }

    console.log('[Cubemap] Created procedural environment cubemap', { size, format });

    return cubemap;
}

/**
 * Create a cubemap view from a cubemap texture
 */
export function createCubemapView(cubemap: GPUTexture): GPUTextureView {
    return cubemap.createView({
        label: 'Environment Cubemap View',
        dimension: 'cube',
        format: cubemap.format,
        mipLevelCount: 1,
        arrayLayerCount: 6,
    });
}

export default createProceduralCubemap;
