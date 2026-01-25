/**
 * DeviceCapabilities - Detect device performance capabilities
 *
 * Uses various browser APIs to determine device tier and
 * appropriate quality settings.
 */

export type DeviceTier = 'low' | 'medium' | 'high' | 'ultra';

export interface CapabilityInfo {
    tier: DeviceTier;
    gpu: string;
    renderer: string;
    webgl2: boolean;
    webgpu: boolean;
    maxTextureSize: number;
    maxRenderbufferSize: number;
    vertexShaderPrecision: string;
    fragmentShaderPrecision: string;
    pointSizeRange: { min: number; max: number };
    maxVertexAttribs: number;
    maxVaryingVectors: number;
    maxVertexTextureImageUnits: number;
    maxFragmentTextureImageUnits: number;
    maxCombinedTextureImageUnits: number;
    memory: number;
    cores: number;
    touch: boolean;
    mobile: boolean;
}

export class DeviceCapabilities {
    private static instance: DeviceCapabilities;
    private info: CapabilityInfo;

    private constructor() {
        this.info = this.detectCapabilities();
    }

    /**
     * Get singleton instance
     */
    public static getInstance(): DeviceCapabilities {
        if (!DeviceCapabilities.instance) {
            DeviceCapabilities.instance = new DeviceCapabilities();
        }
        return DeviceCapabilities.instance;
    }

    /**
     * Detect all device capabilities
     */
    private detectCapabilities(): CapabilityInfo {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');

        if (!gl) {
            console.warn('[DeviceCapabilities] WebGL not supported');
            return this.getDefaultInfo();
        }

        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        const gpu = debugInfo
            ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
            : 'Unknown GPU';

        return {
            tier: this.determineTier(gpu),
            gpu,
            renderer: gpu,
            webgl2: !!canvas.getContext('webgl2'),
            webgpu: 'gpu' in navigator,
            maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
            maxRenderbufferSize: gl.getParameter(gl.MAX_RENDERBUFFER_SIZE),
            vertexShaderPrecision: this.getPrecision(gl, gl.VERTEX_SHADER),
            fragmentShaderPrecision: this.getPrecision(gl, gl.FRAGMENT_SHADER),
            pointSizeRange: gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE),
            maxVertexAttribs: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
            maxVaryingVectors: gl.getParameter(gl.MAX_VARYING_VECTORS),
            maxVertexTextureImageUnits: gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS),
            maxFragmentTextureImageUnits: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS),
            maxCombinedTextureImageUnits: gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS),
            memory: this.estimateMemory(),
            cores: navigator.hardwareConcurrency || 4,
            touch: 'ontouchstart' in window || 'maxTouchPoints' in navigator && navigator.maxTouchPoints > 0,
            mobile: /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
        };
    }

    /**
     * Determine device tier based on GPU
     */
    private determineTier(gpu: string): DeviceTier {
        const gpuLower = gpu.toLowerCase();

        // Ultra tier - High-end desktop GPUs
        if (/(nvidia rtx|nvidia geforce rtx|amd radeon rx \d{4}|amd radeon [78]\d{2}|intel arc|apple m[123])/i.test(gpu)) {
            return 'ultra';
        }

        // High tier - Mid-range to high-end GPUs
        if (/(nvidia gtx|nvidia geforce gtx|amd radeon|intel iris|apple m[12])/i.test(gpu)) {
            return 'high';
        }

        // Medium tier - Integrated GPUs and older cards
        if (/(intel hd|intel uhd|nvidia gtx \d{3}|amd radeon r[57])/i.test(gpu)) {
            return 'medium';
        }

        // Low tier - Older integrated or mobile GPUs
        return 'low';
    }

    /**
     * Get shader precision string
     */
    private getPrecision(gl: WebGLRenderingContext | WebGL2RenderingContext, shaderType: number): string {
        const precision = gl.getShaderPrecisionFormat(shaderType, gl.HIGH_FLOAT);
        if (precision) {
            return `high_${precision.precision}`;
        }
        return 'unknown';
    }

    /**
     * Estimate device memory (rough approximation)
     */
    private estimateMemory(): number {
        // @ts-ignore - deviceMemory is not in all TypeScript definitions
        return navigator.deviceMemory || 4; // Default to 4GB
    }

    /**
     * Get default info for systems without WebGL
     */
    private getDefaultInfo(): CapabilityInfo {
        return {
            tier: 'low',
            gpu: 'Unknown',
            renderer: 'Unknown',
            webgl2: false,
            webgpu: false,
            maxTextureSize: 1024,
            maxRenderbufferSize: 1024,
            vertexShaderPrecision: 'unknown',
            fragmentShaderPrecision: 'unknown',
            pointSizeRange: { min: 1, max: 64 },
            maxVertexAttribs: 8,
            maxVaryingVectors: 8,
            maxVertexTextureImageUnits: 0,
            maxFragmentTextureImageUnits: 8,
            maxCombinedTextureImageUnits: 8,
            memory: 4,
            cores: 4,
            touch: false,
            mobile: false
        };
    }

    /**
     * Get capability info
     */
    public getInfo(): Readonly<CapabilityInfo> {
        return { ...this.info };
    }

    /**
     * Get device tier
     */
    public getTier(): DeviceTier {
        return this.info.tier;
    }

    /**
     * Check if device is mobile
     */
    public isMobile(): boolean {
        return this.info.mobile;
    }

    /**
     * Check if device has touch
     */
    public hasTouch(): boolean {
        return this.info.touch;
    }

    /**
     * Check if WebGL2 is available
     */
    public hasWebGL2(): boolean {
        return this.info.webgl2;
    }

    /**
     * Check if WebGPU is available
     */
    public hasWebGPU(): boolean {
        return this.info.webgpu;
    }

    /**
     * Get recommended particle count
     */
    public getRecommendedParticleCount(): number {
        switch (this.info.tier) {
            case 'ultra': return 50000;
            case 'high': return 25000;
            case 'medium': return 10000;
            case 'low': return 3000;
        }
    }

    /**
     * Get recommended texture size
     */
    public getRecommendedTextureSize(): number {
        switch (this.info.tier) {
            case 'ultra': return 2048;
            case 'high': return 1024;
            case 'medium': return 512;
            case 'low': return 256;
        }
    }

    /**
     * Check if should use high quality shaders
     */
    public useHighQualityShaders(): boolean {
        return this.info.tier === 'high' || this.info.tier === 'ultra';
    }

    /**
     * Check if should use post-processing
     */
    public usePostProcessing(): boolean {
        return this.info.tier === 'high' || this.info.tier === 'ultra';
    }

    /**
     * Log capability info (for debugging)
     */
    public logInfo(): void {
        console.group('[DeviceCapabilities]');
        console.log('Tier:', this.info.tier);
        console.log('GPU:', this.info.gpu);
        console.log('WebGL2:', this.info.webgl2);
        console.log('WebGPU:', this.info.webgpu);
        console.log('Memory:', this.info.memory, 'GB');
        console.log('Cores:', this.info.cores);
        console.log('Mobile:', this.info.mobile);
        console.log('Touch:', this.info.touch);
        console.log('Max Texture Size:', this.info.maxTextureSize);
        console.groupEnd();
    }
}
