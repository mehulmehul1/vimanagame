#!/usr/bin/env node
/**
 * Playwright MCP Server
 * 
 * This server provides browser automation capabilities through the Model Context Protocol.
 * It allows AI assistants to interact with browsers, capture console logs, screenshots, and more.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { chromium, Browser, BrowserContext, Page } from 'playwright';

class PlaywrightMCPServer {
  private server: Server;
  private browser: Browser | null = null;
  private context: BrowserContext | null = null;
  private page: Page | null = null;
  private consoleLogs: Array<{ level: string; text: string; timestamp: Date }> = [];

  constructor() {
    this.server = new Server(
      {
        name: 'playwright-mcp',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupToolHandlers();
    this.setupErrorHandling();
  }

  private setupToolHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: 'launch_browser',
            description: 'Launch a browser instance',
            inputSchema: {
              type: 'object',
              properties: {
                headless: {
                  type: 'boolean',
                  description: 'Run browser in headless mode',
                  default: true,
                },
              },
            },
          },
          {
            name: 'navigate',
            description: 'Navigate to a URL',
            inputSchema: {
              type: 'object',
              properties: {
                url: {
                  type: 'string',
                  description: 'URL to navigate to',
                },
                waitForLoad: {
                  type: 'boolean',
                  description: 'Wait for page to fully load',
                  default: true,
                },
              },
              required: ['url'],
            },
          },
          {
            name: 'get_console_logs',
            description: 'Get captured console logs',
            inputSchema: {
              type: 'object',
              properties: {
                clear: {
                  type: 'boolean',
                  description: 'Clear logs after retrieval',
                  default: false,
                },
                filter: {
                  type: 'string',
                  description: 'Filter by log level (log, error, warn, info, debug)',
                },
              },
            },
          },
          {
            name: 'screenshot',
            description: 'Take a screenshot of the current page',
            inputSchema: {
              type: 'object',
              properties: {
                path: {
                  type: 'string',
                  description: 'Path to save screenshot',
                  default: 'screenshot.png',
                },
                fullPage: {
                  type: 'boolean',
                  description: 'Capture full page or just viewport',
                  default: false,
                },
              },
            },
          },
          {
            name: 'click',
            description: 'Click on an element',
            inputSchema: {
              type: 'object',
              properties: {
                selector: {
                  type: 'string',
                  description: 'CSS selector for the element',
                },
                text: {
                  type: 'string',
                  description: 'Text content to search for (alternative to selector)',
                },
              },
              required: [],
            },
          },
          {
            name: 'type',
            description: 'Type text into an input element',
            inputSchema: {
              type: 'object',
              properties: {
                selector: {
                  type: 'string',
                  description: 'CSS selector for the input',
                },
                text: {
                  type: 'string',
                  description: 'Text to type',
                },
              },
              required: ['selector', 'text'],
            },
          },
          {
            name: 'wait',
            description: 'Wait for a specified duration or condition',
            inputSchema: {
              type: 'object',
              properties: {
                ms: {
                  type: 'number',
                  description: 'Milliseconds to wait',
                },
                selector: {
                  type: 'string',
                  description: 'CSS selector to wait for',
                },
                state: {
                  type: 'string',
                  description: 'State to wait for (visible, hidden, attached, detached)',
                  default: 'visible',
                },
              },
            },
          },
          {
            name: 'close_browser',
            description: 'Close the browser instance',
            inputSchema: {
              type: 'object',
              properties: {},
            },
          },
        ],
      };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case 'launch_browser':
            return await this.launchBrowser(args);
          case 'navigate':
            return await this.navigate(args);
          case 'get_console_logs':
            return await this.getConsoleLogs(args);
          case 'screenshot':
            return await this.takeScreenshot(args);
          case 'click':
            return await this.click(args);
          case 'type':
            return await this.type(args);
          case 'wait':
            return await this.wait(args);
          case 'close_browser':
            return await this.closeBrowser();
          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `Error: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  private async launchBrowser(args: any) {
    this.browser = await chromium.launch({
      headless: args?.headless ?? true,
    });
    this.context = await this.browser.newContext();
    this.page = await this.context.newPage();

    // Set up console log capture
    this.page.on('console', (msg) => {
      this.consoleLogs.push({
        level: msg.type(),
        text: msg.text(),
        timestamp: new Date(),
      });
    });

    // Also capture page errors
    this.page.on('pageerror', (error) => {
      this.consoleLogs.push({
        level: 'error',
        text: `Page Error: ${error.message}`,
        timestamp: new Date(),
      });
    });

    return {
      content: [
        {
          type: 'text',
          text: 'Browser launched successfully with console logging enabled',
        },
      ],
    };
  }

  private async navigate(args: any) {
    if (!this.page) {
      throw new Error('Browser not launched. Call launch_browser first.');
    }

    await this.page.goto(args.url, {
      waitUntil: args?.waitForLoad ? 'networkidle' : 'domcontentloaded',
    });

    return {
      content: [
        {
          type: 'text',
          text: `Navigated to ${args.url}`,
        },
      ],
    };
  }

  private async getConsoleLogs(args: any) {
    let logs = [...this.consoleLogs];

    if (args?.filter) {
      logs = logs.filter((log) => log.level === args.filter);
    }

    const formatted = logs
      .map((log) => `[${log.timestamp.toISOString()}] [${log.level.toUpperCase()}] ${log.text}`)
      .join('\n');

    if (args?.clear) {
      this.consoleLogs = [];
    }

    return {
      content: [
        {
          type: 'text',
          text: formatted || 'No console logs captured',
        },
      ],
    };
  }

  private async takeScreenshot(args: any) {
    if (!this.page) {
      throw new Error('Browser not launched. Call launch_browser first.');
    }

    const path = args?.path || 'screenshot.png';
    await this.page.screenshot({
      path,
      fullPage: args?.fullPage ?? false,
    });

    return {
      content: [
        {
          type: 'text',
          text: `Screenshot saved to ${path}`,
        },
      ],
    };
  }

  private async click(args: any) {
    if (!this.page) {
      throw new Error('Browser not launched. Call launch_browser first.');
    }

    if (args?.selector) {
      await this.page.click(args.selector);
      return {
        content: [
          {
            type: 'text',
            text: `Clicked element: ${args.selector}`,
          },
        ],
      };
    } else if (args?.text) {
      await this.page.getByText(args.text).click();
      return {
        content: [
          {
            type: 'text',
            text: `Clicked element with text: ${args.text}`,
          },
        ],
      };
    } else {
      throw new Error('Either selector or text must be provided');
    }
  }

  private async type(args: any) {
    if (!this.page) {
      throw new Error('Browser not launched. Call launch_browser first.');
    }

    await this.page.fill(args.selector, args.text);

    return {
      content: [
        {
          type: 'text',
          text: `Typed "${args.text}" into ${args.selector}`,
        },
      ],
    };
  }

  private async wait(args: any) {
    if (!this.page) {
      throw new Error('Browser not launched. Call launch_browser first.');
    }

    if (args?.ms) {
      await this.page.waitForTimeout(args.ms);
      return {
        content: [
          {
            type: 'text',
            text: `Waited ${args.ms}ms`,
          },
        ],
      };
    } else if (args?.selector) {
      await this.page.waitForSelector(args.selector, {
        state: args?.state || 'visible',
      });
      return {
        content: [
          {
            type: 'text',
            text: `Waited for selector: ${args.selector}`,
          },
        ],
      };
    } else {
      throw new Error('Either ms or selector must be provided');
    }
  }

  private async closeBrowser() {
    if (this.browser) {
      await this.browser.close();
      this.browser = null;
      this.context = null;
      this.page = null;
      this.consoleLogs = [];
    }

    return {
      content: [
        {
          type: 'text',
          text: 'Browser closed',
        },
      ],
    };
  }

  private setupErrorHandling() {
    this.server.onerror = (error) => {
      console.error('[MCP Error]', error);
    };

    process.on('SIGINT', async () => {
      if (this.browser) {
        await this.browser.close();
      }
      await this.server.close();
      process.exit(0);
    });
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Playwright MCP server running on stdio');
  }
}

// Start the server
const server = new PlaywrightMCPServer();
server.run().catch(console.error);
