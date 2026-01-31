# Playwright MCP Server

A Model Context Protocol (MCP) server that provides browser automation capabilities using Playwright. This server allows AI assistants to launch browsers, navigate to URLs, capture console logs, take screenshots, and interact with web pages.

## Features

- **Browser Automation**: Launch and control browsers (Chromium)
- **Console Log Capture**: Automatically capture all console output (logs, errors, warnings)
- **Screenshots**: Take screenshots of the current page or full page
- **Page Interaction**: Click elements, type text, wait for conditions
- **Error Tracking**: Capture JavaScript errors and page errors

## Installation

```bash
cd mcp-servers/playwright-mcp
npm install
npm run build
```

## Usage with Claude Desktop

Add this to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "playwright": {
      "command": "node",
      "args": [
        "C:/Users/mehul/OneDrive/Desktop/Studio/PROJECTS/shadowczarengine/mcp-servers/playwright-mcp/dist/server.js"
      ]
    }
  }
}
```

## Available Tools

### 1. launch_browser
Launch a browser instance with console logging enabled.

**Parameters:**
- `headless` (boolean, optional): Run in headless mode (default: true)

### 2. navigate
Navigate to a URL.

**Parameters:**
- `url` (string, required): URL to navigate to
- `waitForLoad` (boolean, optional): Wait for page to fully load (default: true)

### 3. get_console_logs
Retrieve captured console logs.

**Parameters:**
- `clear` (boolean, optional): Clear logs after retrieval (default: false)
- `filter` (string, optional): Filter by log level (log, error, warn, info, debug)

### 4. screenshot
Take a screenshot of the current page.

**Parameters:**
- `path` (string, optional): Path to save screenshot (default: 'screenshot.png')
- `fullPage` (boolean, optional): Capture full page (default: false)

### 5. click
Click on an element.

**Parameters:**
- `selector` (string, optional): CSS selector for the element
- `text` (string, optional): Text content to search for

### 6. type
Type text into an input element.

**Parameters:**
- `selector` (string, required): CSS selector for the input
- `text` (string, required): Text to type

### 7. wait
Wait for a specified duration or condition.

**Parameters:**
- `ms` (number, optional): Milliseconds to wait
- `selector` (string, optional): CSS selector to wait for
- `state` (string, optional): State to wait for (visible, hidden, attached, detached)

### 8. close_browser
Close the browser instance.

## Example Workflow

```
1. Launch browser with headless=false to see the window
2. Navigate to http://localhost:5173
3. Wait 3000ms for the app to load
4. Get console logs to check for errors
5. Take a screenshot to verify the UI
6. Close browser
```

## Checking Console Logs

The server automatically captures:
- `console.log()` messages
- `console.error()` messages  
- `console.warn()` messages
- `console.info()` messages
- Page JavaScript errors

Use `get_console_logs` with `filter: "error"` to see only errors, or without filter to see all logs with timestamps.

## Development

```bash
# Watch mode for development
npm run dev

# Build for production
npm run build
```
