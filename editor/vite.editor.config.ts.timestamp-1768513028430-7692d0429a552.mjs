// vite.editor.config.ts
import { defineConfig } from "file:///C:/Users/mehul/OneDrive/Desktop/Studio/PROJECTS/shadowczarengine/editor/node_modules/vite/dist/node/index.js";
import react from "file:///C:/Users/mehul/OneDrive/Desktop/Studio/PROJECTS/shadowczarengine/editor/node_modules/@vitejs/plugin-react/dist/index.js";
import topLevelAwait from "file:///C:/Users/mehul/OneDrive/Desktop/Studio/PROJECTS/shadowczarengine/editor/node_modules/vite-plugin-top-level-await/exports/import.mjs";
import wasm from "file:///C:/Users/mehul/OneDrive/Desktop/Studio/PROJECTS/shadowczarengine/node_modules/vite-plugin-wasm/exports/import.mjs";
import { resolve } from "path";
var __vite_injected_original_dirname = "C:\\Users\\mehul\\OneDrive\\Desktop\\Studio\\PROJECTS\\shadowczarengine\\editor";
var vite_editor_config_default = defineConfig({
  plugins: [
    react(),
    wasm(),
    topLevelAwait({
      // The Promise name that will be used in the code
      promiseExportName: "tla",
      // The maximum number of dynamic imports that can be awaited
      promiseImportName: "tlaImport"
    })
  ],
  resolve: {
    alias: {
      "@": resolve(__vite_injected_original_dirname, "./"),
      "@/core": resolve(__vite_injected_original_dirname, "./core"),
      "@/panels": resolve(__vite_injected_original_dirname, "./panels"),
      "@/components": resolve(__vite_injected_original_dirname, "./components"),
      "@/data": resolve(__vite_injected_original_dirname, "./data"),
      "@/utils": resolve(__vite_injected_original_dirname, "./utils"),
      "@/styles": resolve(__vite_injected_original_dirname, "./styles")
    }
  },
  server: {
    port: 3001,
    strictPort: false,
    host: true,
    open: true,
    headers: {
      "Access-Control-Allow-Origin": "*"
    },
    // Reduce header size to avoid 431 errors
    fs: {
      strict: false
    }
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          "react-vendor": ["react", "react-dom"],
          "pcui-vendor": ["@playcanvas/pcui", "@playcanvas/observer"],
          "three-vendor": ["three"]
        }
      }
    }
  },
  optimizeDeps: {
    include: ["react", "react-dom", "@playcanvas/pcui", "@playcanvas/observer", "three"]
  }
});
export {
  vite_editor_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5lZGl0b3IuY29uZmlnLnRzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfZGlybmFtZSA9IFwiQzpcXFxcVXNlcnNcXFxcbWVodWxcXFxcT25lRHJpdmVcXFxcRGVza3RvcFxcXFxTdHVkaW9cXFxcUFJPSkVDVFNcXFxcc2hhZG93Y3phcmVuZ2luZVxcXFxlZGl0b3JcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfZmlsZW5hbWUgPSBcIkM6XFxcXFVzZXJzXFxcXG1laHVsXFxcXE9uZURyaXZlXFxcXERlc2t0b3BcXFxcU3R1ZGlvXFxcXFBST0pFQ1RTXFxcXHNoYWRvd2N6YXJlbmdpbmVcXFxcZWRpdG9yXFxcXHZpdGUuZWRpdG9yLmNvbmZpZy50c1wiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9pbXBvcnRfbWV0YV91cmwgPSBcImZpbGU6Ly8vQzovVXNlcnMvbWVodWwvT25lRHJpdmUvRGVza3RvcC9TdHVkaW8vUFJPSkVDVFMvc2hhZG93Y3phcmVuZ2luZS9lZGl0b3Ivdml0ZS5lZGl0b3IuY29uZmlnLnRzXCI7aW1wb3J0IHsgZGVmaW5lQ29uZmlnIH0gZnJvbSAndml0ZSc7XHJcbmltcG9ydCByZWFjdCBmcm9tICdAdml0ZWpzL3BsdWdpbi1yZWFjdCc7XHJcbmltcG9ydCB0b3BMZXZlbEF3YWl0IGZyb20gJ3ZpdGUtcGx1Z2luLXRvcC1sZXZlbC1hd2FpdCc7XHJcbmltcG9ydCB3YXNtIGZyb20gJ3ZpdGUtcGx1Z2luLXdhc20nO1xyXG5pbXBvcnQgeyByZXNvbHZlIH0gZnJvbSAncGF0aCc7XHJcblxyXG4vLyBodHRwczovL3ZpdGVqcy5kZXYvY29uZmlnL1xyXG5leHBvcnQgZGVmYXVsdCBkZWZpbmVDb25maWcoe1xyXG4gIHBsdWdpbnM6IFtcclxuICAgIHJlYWN0KCksXHJcbiAgICB3YXNtKCksXHJcbiAgICB0b3BMZXZlbEF3YWl0KHtcclxuICAgICAgLy8gVGhlIFByb21pc2UgbmFtZSB0aGF0IHdpbGwgYmUgdXNlZCBpbiB0aGUgY29kZVxyXG4gICAgICBwcm9taXNlRXhwb3J0TmFtZTogJ3RsYScsXHJcbiAgICAgIC8vIFRoZSBtYXhpbXVtIG51bWJlciBvZiBkeW5hbWljIGltcG9ydHMgdGhhdCBjYW4gYmUgYXdhaXRlZFxyXG4gICAgICBwcm9taXNlSW1wb3J0TmFtZTogJ3RsYUltcG9ydCdcclxuICAgIH0pXHJcbiAgXSxcclxuICByZXNvbHZlOiB7XHJcbiAgICBhbGlhczoge1xyXG4gICAgICAnQCc6IHJlc29sdmUoX19kaXJuYW1lLCAnLi8nKSxcclxuICAgICAgJ0AvY29yZSc6IHJlc29sdmUoX19kaXJuYW1lLCAnLi9jb3JlJyksXHJcbiAgICAgICdAL3BhbmVscyc6IHJlc29sdmUoX19kaXJuYW1lLCAnLi9wYW5lbHMnKSxcclxuICAgICAgJ0AvY29tcG9uZW50cyc6IHJlc29sdmUoX19kaXJuYW1lLCAnLi9jb21wb25lbnRzJyksXHJcbiAgICAgICdAL2RhdGEnOiByZXNvbHZlKF9fZGlybmFtZSwgJy4vZGF0YScpLFxyXG4gICAgICAnQC91dGlscyc6IHJlc29sdmUoX19kaXJuYW1lLCAnLi91dGlscycpLFxyXG4gICAgICAnQC9zdHlsZXMnOiByZXNvbHZlKF9fZGlybmFtZSwgJy4vc3R5bGVzJylcclxuICAgIH1cclxuICB9LFxyXG4gIHNlcnZlcjoge1xyXG4gICAgcG9ydDogMzAwMSxcclxuICAgIHN0cmljdFBvcnQ6IGZhbHNlLFxyXG4gICAgaG9zdDogdHJ1ZSxcclxuICAgIG9wZW46IHRydWUsXHJcbiAgICBoZWFkZXJzOiB7XHJcbiAgICAgICdBY2Nlc3MtQ29udHJvbC1BbGxvdy1PcmlnaW4nOiAnKidcclxuICAgIH0sXHJcbiAgICAvLyBSZWR1Y2UgaGVhZGVyIHNpemUgdG8gYXZvaWQgNDMxIGVycm9yc1xyXG4gICAgZnM6IHtcclxuICAgICAgc3RyaWN0OiBmYWxzZVxyXG4gICAgfVxyXG4gIH0sXHJcbiAgYnVpbGQ6IHtcclxuICAgIG91dERpcjogJ2Rpc3QnLFxyXG4gICAgZW1wdHlPdXREaXI6IHRydWUsXHJcbiAgICBzb3VyY2VtYXA6IHRydWUsXHJcbiAgICByb2xsdXBPcHRpb25zOiB7XHJcbiAgICAgIG91dHB1dDoge1xyXG4gICAgICAgIG1hbnVhbENodW5rczoge1xyXG4gICAgICAgICAgJ3JlYWN0LXZlbmRvcic6IFsncmVhY3QnLCAncmVhY3QtZG9tJ10sXHJcbiAgICAgICAgICAncGN1aS12ZW5kb3InOiBbJ0BwbGF5Y2FudmFzL3BjdWknLCAnQHBsYXljYW52YXMvb2JzZXJ2ZXInXSxcclxuICAgICAgICAgICd0aHJlZS12ZW5kb3InOiBbJ3RocmVlJ11cclxuICAgICAgICB9XHJcbiAgICAgIH1cclxuICAgIH1cclxuICB9LFxyXG4gIG9wdGltaXplRGVwczoge1xyXG4gICAgaW5jbHVkZTogWydyZWFjdCcsICdyZWFjdC1kb20nLCAnQHBsYXljYW52YXMvcGN1aScsICdAcGxheWNhbnZhcy9vYnNlcnZlcicsICd0aHJlZSddXHJcbiAgfVxyXG59KTtcclxuIl0sCiAgIm1hcHBpbmdzIjogIjtBQUF1YSxTQUFTLG9CQUFvQjtBQUNwYyxPQUFPLFdBQVc7QUFDbEIsT0FBTyxtQkFBbUI7QUFDMUIsT0FBTyxVQUFVO0FBQ2pCLFNBQVMsZUFBZTtBQUp4QixJQUFNLG1DQUFtQztBQU96QyxJQUFPLDZCQUFRLGFBQWE7QUFBQSxFQUMxQixTQUFTO0FBQUEsSUFDUCxNQUFNO0FBQUEsSUFDTixLQUFLO0FBQUEsSUFDTCxjQUFjO0FBQUE7QUFBQSxNQUVaLG1CQUFtQjtBQUFBO0FBQUEsTUFFbkIsbUJBQW1CO0FBQUEsSUFDckIsQ0FBQztBQUFBLEVBQ0g7QUFBQSxFQUNBLFNBQVM7QUFBQSxJQUNQLE9BQU87QUFBQSxNQUNMLEtBQUssUUFBUSxrQ0FBVyxJQUFJO0FBQUEsTUFDNUIsVUFBVSxRQUFRLGtDQUFXLFFBQVE7QUFBQSxNQUNyQyxZQUFZLFFBQVEsa0NBQVcsVUFBVTtBQUFBLE1BQ3pDLGdCQUFnQixRQUFRLGtDQUFXLGNBQWM7QUFBQSxNQUNqRCxVQUFVLFFBQVEsa0NBQVcsUUFBUTtBQUFBLE1BQ3JDLFdBQVcsUUFBUSxrQ0FBVyxTQUFTO0FBQUEsTUFDdkMsWUFBWSxRQUFRLGtDQUFXLFVBQVU7QUFBQSxJQUMzQztBQUFBLEVBQ0Y7QUFBQSxFQUNBLFFBQVE7QUFBQSxJQUNOLE1BQU07QUFBQSxJQUNOLFlBQVk7QUFBQSxJQUNaLE1BQU07QUFBQSxJQUNOLE1BQU07QUFBQSxJQUNOLFNBQVM7QUFBQSxNQUNQLCtCQUErQjtBQUFBLElBQ2pDO0FBQUE7QUFBQSxJQUVBLElBQUk7QUFBQSxNQUNGLFFBQVE7QUFBQSxJQUNWO0FBQUEsRUFDRjtBQUFBLEVBQ0EsT0FBTztBQUFBLElBQ0wsUUFBUTtBQUFBLElBQ1IsYUFBYTtBQUFBLElBQ2IsV0FBVztBQUFBLElBQ1gsZUFBZTtBQUFBLE1BQ2IsUUFBUTtBQUFBLFFBQ04sY0FBYztBQUFBLFVBQ1osZ0JBQWdCLENBQUMsU0FBUyxXQUFXO0FBQUEsVUFDckMsZUFBZSxDQUFDLG9CQUFvQixzQkFBc0I7QUFBQSxVQUMxRCxnQkFBZ0IsQ0FBQyxPQUFPO0FBQUEsUUFDMUI7QUFBQSxNQUNGO0FBQUEsSUFDRjtBQUFBLEVBQ0Y7QUFBQSxFQUNBLGNBQWM7QUFBQSxJQUNaLFNBQVMsQ0FBQyxTQUFTLGFBQWEsb0JBQW9CLHdCQUF3QixPQUFPO0FBQUEsRUFDckY7QUFDRixDQUFDOyIsCiAgIm5hbWVzIjogW10KfQo=
