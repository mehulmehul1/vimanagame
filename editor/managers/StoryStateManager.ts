import type {
  StoryData,
  StoryState,
  Transition,
  StoryAction,
} from '../types/story';
import { loadStoryData, saveStoryData } from '../utils/storyDataIO';

/**
 * StoryStateManager - Central data manager for story editing
 */
class StoryDataManager {
  private data: StoryData;
  private listeners: Set<(data: StoryData) => void>;
  private undoStack: StoryAction[];
  private redoStack: StoryAction[];
  private filePath: string | null;
  private maxUndoStack: number = 100;

  constructor(initialData?: StoryData) {
    this.data = initialData || this.getEmptyData();
    this.listeners = new Set();
    this.undoStack = [];
    this.redoStack = [];
    this.filePath = null;
  }

  private getEmptyData(): StoryData {
    return {
      states: {},
      acts: {},
    };
  }

  async load(filePath: string): Promise<StoryData> {
    const data = await loadStoryData(filePath);
    this.data = data;
    this.filePath = filePath;
    this.notifyListeners();
    return this.data;
  }

  async save(filePath?: string): Promise<void> {
    const targetPath = filePath || this.filePath;
    if (!targetPath) {
      throw new Error('No file path specified');
    }
    // Browser-safe filename extraction
    const filename = targetPath.split('/').pop() || 'storyData.json';
    await saveStoryData(this.data, filename);
    this.filePath = targetPath;
  }

  getData(): StoryData {
    return this.data;
  }

  getState(id: string): StoryState | undefined {
    return this.data.states[id];
  }

  addState(state: StoryState): void {
    this.data.states[state.id] = state;
    this.notifyListeners();
  }

  updateState(id: string, updates: Partial<StoryState>): void {
    this.data.states[id] = { ...this.data.states[id], ...updates };
    this.notifyListeners();
  }

  deleteState(id: string): void {
    delete this.data.states[id];
    this.notifyListeners();
  }

  duplicateState(id: string, newId: string, newLabel: string): StoryState {
    const original = this.data.states[id];
    if (!original) {
      throw new Error('State ' + id + ' not found');
    }
    const duplicate: StoryState = {
      ...original,
      id: newId,
      label: newLabel,
      value: this.getNextStateValue(),
      transitions: original.transitions.map(t => ({
        ...t,
        id: t.id + '_copy',
        from: t.from === id ? newId : t.from,
      })),
    };
    this.addState(duplicate);
    return duplicate;
  }

  addTransition(fromId: string, toId: string, transition: Omit<Transition, 'id'>): Transition {
    const newTransition: Transition = {
      ...transition,
      id: 'trans_' + fromId + '_' + toId + '_' + Date.now(),
      from: fromId,
      to: toId,
    };
    const state = this.data.states[fromId];
    if (!state) {
      throw new Error('State ' + fromId + ' not found');
    }
    state.transitions.push(newTransition);
    this.notifyListeners();
    return newTransition;
  }

  deleteTransition(transitionId: string): void {
    for (const [stateId, state] of Object.entries(this.data.states)) {
      const index = state.transitions.findIndex(t => t.id === transitionId);
      if (index !== -1) {
        state.transitions.splice(index, 1);
        this.notifyListeners();
        return;
      }
    }
  }

  subscribe(callback: (data: StoryData) => void): () => void {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  private notifyListeners(): void {
    this.listeners.forEach(cb => cb(this.data));
  }

  undo(): StoryAction | undefined {
    const action = this.undoStack.pop();
    if (!action) return undefined;
    this.redoStack.push(action);
    return action;
  }

  redo(): StoryAction | undefined {
    const action = this.redoStack.pop();
    if (!action) return undefined;
    this.undoStack.push(action);
    return action;
  }

  canUndo(): boolean {
    return this.undoStack.length > 0;
  }

  canRedo(): boolean {
    return this.redoStack.length > 0;
  }

  getNextStateValue(): number {
    const values = Object.values(this.data.states).map(s => s.value).filter(v => typeof v === 'number');
    return values.length > 0 ? Math.max(...values) + 1 : 0;
  }

  clear(): void {
    this.undoStack = [];
    this.redoStack = [];
  }
}

let instance: StoryDataManager | null = null;

export function getStoryDataManager(): StoryDataManager {
  if (!instance) {
    instance = new StoryDataManager();
  }
  return instance;
}

export default StoryDataManager;
