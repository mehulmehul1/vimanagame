/**
 * story.ts - TypeScript interfaces for Story Data Structure
 * =============================================================================
 *
 * Complete type definitions for the story-centric scene flow system.
 * Defines states, transitions, acts, and all narrative flow data.
 */

/**
 * Trigger types for state transitions
 */
export type TriggerType =
  | "onComplete"       // Video/dialog finishes naturally
  | "onChoice"         // Player selects a dialog option
  | "onTimeout"        // Timer expires
  | "onProximity"      // Player enters/leaves an area
  | "onInteract"       // Player clicks/hovers an object
  | "onState"          // Another state reaches a value
  | "custom";          // Custom JavaScript condition

/**
 * Position data for player spawn/debug
 */
export interface PlayerPosition {
  x: number;
  y: number;
  z: number;
  rotation?: {
    x: number;
    y: number;
    z: number;
  };
}

/**
 * Content attachments for a state
 */
export interface StateContent {
  video?: string;              // Path to video file
  dialog?: string;             // Dialog tree ID
  music?: string;              // Music track ID
  sfx?: string[];              // SFX IDs
  cameraAnimation?: string;    // Camera animation ID
}

/**
 * Entry criteria for when a state becomes active
 */
export interface EntryCriteria {
  currentState?: {
    $gte?: number;            // Greater than or equal
    $lt?: number;             // Less than
    $in?: number[];           // In array
  };
  dialogChoice?: number;      // For branching on choices
  customCondition?: string;   // JavaScript expression
}

/**
 * Transition between states
 */
export interface Transition {
  id: string;                 // Unique transition ID
  from: string;               // Source state ID
  to: string;                 // Target state ID
  trigger: TriggerType;
  label: string;              // Human-readable description

  // Trigger-specific data
  condition?: string;         // JavaScript expression for custom triggers
  dialogChoice?: number;      // Which choice leads here
  timeout?: number;           // ms before auto-transition
  proximity?: {               // Position-based trigger
    x: number;
    y: number;
    z: number;
    radius: number;
  };
}

/**
 * Complete story state definition
 */
export interface StoryState {
  id: string;                 // e.g., "PHONE_BOOTH_RINGING"
  value: number;              // e.g., 6
  label: string;              // e.g., "Phone Booth Ringing"
  act: number;                // 1, 2, 3...
  category: string;           // Matches STATE_CATEGORIES
  color: string;              // For UI

  // Physical context
  zone: string | null;        // e.g., "plaza"
  playerPosition?: PlayerPosition;

  // Content attachments
  content: StateContent;

  // Entry criteria (when this state becomes active)
  criteria?: EntryCriteria;

  // Outgoing transitions
  transitions: Transition[];

  // Metadata
  description?: string;
  notes?: string;
}

/**
 * Act grouping definition
 */
export interface Act {
  name: string;               // e.g., "Act 1: Introduction"
  color: string;              // For UI theming
  states: string[];           // State IDs in this act
}

/**
 * Complete story data structure
 */
export interface StoryData {
  states: Record<string, StoryState>;
  acts: Record<string, Act>;
}

/**
 * Zone information (reference to scene manager zones)
 */
export interface ZoneInfo {
  id: string;
  name: string;
  color?: string;
}

/**
 * Dialog tree reference
 */
export interface DialogReference {
  id: string;
  name: string;
  choiceCount: number;
}

/**
 * Video asset reference
 */
export interface VideoReference {
  id: string;
  name: string;
  path: string;
  duration?: number;
}

/**
 * Music track reference
 */
export interface MusicReference {
  id: string;
  name: string;
  path: string;
  loop: boolean;
}

/**
 * State inspector panel data
 */
export interface StateInspectorData {
  state: StoryState | null;
  availableZones: ZoneInfo[];
  availableDialogs: DialogReference[];
  availableVideos: VideoReference[];
  availableMusic: MusicReference[];
}

/**
 * Undo/Redo action types
 */
export type StoryActionType =
  | "ADD_STATE"
  | "DELETE_STATE"
  | "UPDATE_STATE"
  | "DUPLICATE_STATE"
  | "ADD_TRANSITION"
  | "DELETE_TRANSITION"
  | "UPDATE_TRANSITION";

/**
 * Undo/Redo action record
 */
export interface StoryAction {
  type: StoryActionType;
  timestamp: number;
  description: string;

  // State snapshot before action
  before: {
    stateId?: string;
    stateData?: StoryState;
    transitionId?: string;
    transitionData?: Transition;
  };

  // State snapshot after action
  after: {
    stateId?: string;
    stateData?: StoryState;
    transitionId?: string;
    transitionData?: Transition;
  };
}

/**
 * Story state manager events
 */
export type StoryEventManager = {
  onStateChange?: (stateId: string, state: StoryState) => void;
  onStateAdd?: (stateId: string, state: StoryState) => void;
  onStateDelete?: (stateId: string) => void;
  onTransitionAdd?: (fromId: string, toId: string, transition: Transition) => void;
  onTransitionDelete?: (transitionId: string) => void;
  onDataChange?: (data: StoryData) => void;
};

export default StoryData;
