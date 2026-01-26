# Dialog Choice UI - First Principles Guide

## Overview

The **Dialog Choice UI** (also called Conversation System) is how the game presents narrative content and lets players make decisions that affect the story. It displays character dialogue, shows player response options, and captures player choices that branch the narrative. This system is essential for storytelling games, RPGs, and any game with interactive conversations.

Think of the dialog system as the **"interactive narrative"** engine—like a choose-your-own-adventure book brought to life, it presents story content and lets players shape the direction through their choices.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Make players feel like their choices matter. The dialog system should present compelling choices, respond meaningfully to player decisions, and create emotional investment in the story's outcome.

**Why a Dialog Choice System?**
- **Player Agency**: Choices make players active participants, not observers
- **Narrative Branching**: Different paths create replayability
- **Character Expression**: Dialogue reveals personality and world
- **Emotional Engagement**: Players care about outcomes they influenced
- **Pacing**: Control information flow and build tension

**Player Psychology**:
```
Dialog Appears → "What's happening?" → Attention
     ↓
Read Dialogue → "I understand" → Comprehension
     ↓
See Choices → "I can decide" → Agency
     ↓
Make Choice → "I did that" → Ownership
     ↓
Story Responds → "My choice mattered!" → Investment
     ↓
Remember Choice → "I wonder what if..." → Speculation
```

### Design Principles

**1. Clear Presentation**
Dialog should be easy to read:
- Readable fonts with good contrast
- Appropriate pacing (not too fast/slow)
- Speaker identification clear
- Text color/style differentiates narration vs. speech

**2. Meaningful Choices**
Every choice should feel like it matters:
- Avoid "illusion of choice" (same outcome regardless)
- Consequences should be clear or appropriately mysterious
- Some choices immediate, some long-term
- Include personality options (how you say it matters too)

**3. Responsive Feedback**
The world should react to choices:
- Characters remember what you said
- Later references to past choices
- Different dialogue based on relationship
- Visible consequences when appropriate

**4. Pacing Variety**
Mix up dialog styles:
- Quick back-and-forth conversations
- Long monologues for important info
- Silent moments for emotional impact
- Action interrupts during dialog

### Dialog Structure

A typical dialog node structure:

```
DIALOG NODE STRUCTURE:

┌─────────────────────────────────────────────────────────┐
│  DIALOG NODE                                            │
│  ├── speakerId: "character_nickname"                    │
│  ├── text: "Dialog text here"                           │
│  ├── expression: "neutral" (affects portrait/animation)  │
│  ├── voiceLine: "audio_file_path" (optional)            │
│  ├── action: "animation_name" (optional)                │
│  └── choices: [                                         │
│       {                                                 │
│         text: "Choice 1",                               │
│         nextNode: "dialog_node_2",                      │
│         condition: "met_character_before",              │
│         consequences: ["relationship:+1"]                │
│       },                                                │
│       {                                                 │
│         text: "Choice 2",                               │
│         nextNode: "dialog_node_3",                      │
│         consequences: ["relationship:-1", "trust:-1"]    │
│       }                                                 │
│     ]                                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the dialog system, you should know:
- **Tree data structures** - Dialog branching is a tree/graph
- **State management** - Tracking dialog history and flags
- **Text timing** - Typewriter effects, pacing
- **Conditionals** - Showing/hiding choices based on game state
- **Event emission** - Notifying other systems of choices

### Core Architecture

```
DIALOG SYSTEM ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                  DIALOG MANAGER                         │
│  - Current dialog tracking                              │
│  - History and flags                                    │
│  - Typewriter/pacing control                            │
│  - Choice condition evaluation                          │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   UI DISPLAY  │  │   DATA       │  │   EVENTS     │
│  - Dialog box │  │  - Dialog    │  │  - On choice │
│  - Portrait   │  │    tree      │  │  - On complete│
│  - Choices    │  │  - Condition │  │  - Consequence │
└───────────────┘  └───────────────┘  └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   STORAGE    │
                    │  - Flags     │
                    │  - History   │
                    │  - State     │
                    └───────────────┘
```

### DialogManager Class

```javascript
class DialogManager {
  constructor(options = {}) {
    this.gameManager = options.gameManager;
    this.uiManager = options.uiManager;
    this.logger = options.logger || console;

    // Dialog state
    this.isActive = false;
    this.currentDialog = null;
    this.currentNode = null;
    this.dialogHistory = [];
    this.flags = {};  // Set from dialog choices

    // Configuration
    this.config = {
      typewriterEffect: true,
      typewriterSpeed: 30,  // ms per character
      autoAdvance: false,
      autoAdvanceDelay: 3000,
      showPortraits: true,
      skipButton: true,
      voiceVolume: 1.0
    };

    Object.assign(this.config, options.config || {});

    // Typewriter state
    this.typewriterTimeout = null;
    this.currentText = '';
    this.targetText = '';
    this.charIndex = 0;
    this.isTyping = false;
    this.canSkip = true;

    // Setup UI
    this.setupUI();
  }

  /**
   * Setup dialog UI elements
   */
  setupUI() {
    // Create dialog container
    this.container = document.createElement('div');
    this.container.className = 'dialog-container';
    this.container.innerHTML = `
      <div class="dialog-box">
        <!-- Speaker portrait -->
        <div class="dialog-portrait" style="display: none;">
          <img class="portrait-image" src="" alt="">
          <div class="portrait-name"></div>
        </div>

        <!-- Dialog content -->
        <div class="dialog-content">
          <div class="dialog-speaker"></div>
          <div class="dialog-text">
            <span class="text-content"></span>
            <span class="text-cursor">▋</span>
          </div>
        </div>

        <!-- Choices container -->
        <div class="dialog-choices"></div>

        <!-- Dialog controls -->
        <div class="dialog-controls">
          <button class="skip-btn" style="display: none;">Skip ▐▐</button>
          <button class="continue-btn" style="display: none;">Continue ▼</button>
        </div>
      </div>

      <!-- Dialog history (for review) -->
      <div class="dialog-history-panel" style="display: none;">
        <div class="history-header">Conversation History</div>
        <div class="history-content"></div>
      </div>
    `;

    // Store element references
    this.dialogBox = this.container.querySelector('.dialog-box');
    this.portraitElement = this.container.querySelector('.dialog-portrait');
    this.portraitImage = this.container.querySelector('.portrait-image');
    this.portraitName = this.container.querySelector('.portrait-name');
    this.speakerElement = this.container.querySelector('.dialog-speaker');
    this.textElement = this.container.querySelector('.text-content');
    this.textCursor = this.container.querySelector('.text-cursor');
    this.choicesElement = this.container.querySelector('.dialog-choices');
    this.skipButton = this.container.querySelector('.skip-btn');
    this.continueButton = this.container.querySelector('.continue-btn');

    // Setup event listeners
    this.setupEventListeners();

    // Add to UI container
    this.uiManager.overlayContainer.appendChild(this.container);
  }

  /**
   * Setup event listeners
   */
  setupEventListeners() {
    // Skip button - finish typewriter effect immediately
    this.skipButton.addEventListener('click', () => {
      if (this.isTyping && this.canSkip) {
        this.finishTypewriter();
      }
    });

    // Continue button - advance to next or select first choice
    this.continueButton.addEventListener('click', () => {
      if (this.currentNode && this.currentNode.choices && this.currentNode.choices.length > 0) {
        // Select first choice
        this.selectChoice(0);
      } else {
        this.advance();
      }
    });

    // Click to advance (when not clicking choices)
    this.dialogBox.addEventListener('click', (e) => {
      if (e.target === this.dialogBox || e.target.closest('.dialog-text')) {
        if (this.isTyping && this.canSkip) {
          this.finishTypewriter();
        } else if (this.continueButton.style.display !== 'none') {
          this.continueButton.click();
        }
      }
    });

    // Keyboard input
    document.addEventListener('keydown', (e) => this.onKeyDown(e));
  }

  /**
   * Handle keyboard input
   */
  onKeyDown(event) {
    if (!this.isActive) return;

    switch (event.key) {
      case 'Enter':
      case ' ':
        event.preventDefault();
        if (this.isTyping && this.canSkip) {
          this.finishTypewriter();
        } else if (this.continueButton.style.display !== 'none') {
          this.continueButton.click();
        }
        break;

      case 'Escape':
        event.preventDefault();
        this.endDialog();
        break;

      case '1':
      case '2':
      case '3':
      case '4':
        // Number keys for choices
        const choiceIndex = parseInt(event.key) - 1;
        if (choiceIndex < this.currentNode?.choices?.length) {
          this.selectChoice(choiceIndex);
        }
        break;
    }
  }

  /**
   * Start a dialog
   */
  startDialog(dialogId, startNode = 'start') {
    // Load dialog data
    this.currentDialog = this.loadDialog(dialogId);
    if (!this.currentDialog) {
      this.logger.error(`Dialog "${dialogId}" not found`);
      return;
    }

    // Reset state
    this.isActive = true;
    this.dialogHistory = [];
    this.container.style.display = '';
    this.container.classList.add('active');

    // Start at specified node
    this.goToNode(startNode);

    // Emit event
    this.gameManager.emit('dialog:started', { dialogId });
  }

  /**
   * Go to a specific dialog node
   */
  goToNode(nodeId) {
    const node = this.currentDialog.nodes[nodeId];
    if (!node) {
      this.logger.warn(`Dialog node "${nodeId}" not found`);
      this.endDialog();
      return;
    }

    this.currentNode = node;

    // Check node conditions
    if (node.condition && !this.evaluateCondition(node.condition)) {
      // Skip to fallback node
      if (node.fallbackNode) {
        this.goToNode(node.fallbackNode);
        return;
      }
    }

    // Display node
    this.displayNode(node);

    // Add to history
    this.dialogHistory.push({
      nodeId,
      speaker: node.speakerId,
      text: node.text
    });

    // Emit event
    this.gameManager.emit('dialog:nodeShown', { nodeId, node });
  }

  /**
   * Display a dialog node
   */
  displayNode(node) {
    // Hide choices and continue button initially
    this.choicesElement.innerHTML = '';
    this.choicesElement.style.display = 'none';
    this.continueButton.style.display = 'none';

    // Update speaker
    const speaker = this.getSpeakerInfo(node.speakerId);
    this.speakerElement.textContent = speaker?.displayName || node.speakerId || '';

    // Update portrait
    if (this.config.showPortraits && speaker?.portrait) {
      this.portraitElement.style.display = '';
      this.portraitImage.src = speaker.portrait;
      this.portraitName.textContent = speaker?.displayName || '';
      this.portraitImage.className = `portrait-image expression-${node.expression || 'neutral'}`;
    } else {
      this.portraitElement.style.display = 'none';
    }

    // Start typewriter effect for text
    this.displayText(node.text);

    // Play voice line if present
    if (node.voiceLine) {
      this.playVoiceLine(node.voiceLine);
    }

    // Trigger action if present
    if (node.action) {
      this.gameManager.emit('dialog:action', { action: node.action });
    }

    // Check if there are choices
    const hasChoices = node.choices && node.choices.length > 0;
    const hasNext = node.nextNode || node.nextNode === null;  // null = end

    if (!hasChoices && !hasNext) {
      // End of dialog, will auto-advance
      return;
    }

    // Show continue button or choices after text completes
    const showOptions = () => {
      if (hasChoices) {
        this.showChoices(node.choices);
      } else if (hasNext) {
        this.continueButton.style.display = '';
      }
    };

    if (this.isTyping) {
      this.onTypewriterComplete = showOptions;
    } else {
      showOptions();
    }

    // Auto-advance if configured
    if (this.config.autoAdvance && !hasChoices) {
      setTimeout(() => {
        if (this.currentNode === node) {
          this.advance();
        }
      }, this.config.autoAdvanceDelay);
    }
  }

  /**
   * Display text with typewriter effect
   */
  displayText(text) {
    this.targetText = text;
    this.charIndex = 0;
    this.currentText = '';
    this.isTyping = true;

    // Show skip button
    if (this.config.skipButton) {
      this.skipButton.style.display = '';
    }

    // Clear and show cursor
    this.textElement.textContent = '';
    this.textCursor.style.display = 'inline';

    if (this.config.typewriterEffect) {
      this.startTypewriter();
    } else {
      this.textElement.textContent = text;
      this.isTyping = false;
      this.skipButton.style.display = 'none';
      this.textCursor.style.display = 'none';
    }
  }

  /**
   * Start typewriter effect
   */
  startTypewriter() {
    const typeChar = () => {
      if (this.charIndex >= this.targetText.length) {
        this.isTyping = false;
        this.skipButton.style.display = 'none';
        this.textCursor.style.display = 'none';

        if (this.onTypewriterComplete) {
          this.onTypewriterComplete();
          this.onTypewriterComplete = null;
        }
        return;
      }

      // Add next character
      this.charIndex++;
      this.currentText = this.targetText.substring(0, this.charIndex);
      this.textElement.textContent = this.currentText;

      // Play sound (subtle)
      if (this.charIndex % 3 === 0) {  // Every 3rd character
        this.gameManager.audioManager.playSfx('dialog_blip', {
          volume: 0.1,
          pitch: 0.8 + Math.random() * 0.4
        });
      }

      // Schedule next character
      this.typewriterTimeout = setTimeout(typeChar, this.config.typewriterSpeed);
    };

    typeChar();
  }

  /**
   * Finish typewriter effect immediately
   */
  finishTypewriter() {
    if (this.typewriterTimeout) {
      clearTimeout(this.typewriterTimeout);
      this.typewriterTimeout = null;
    }

    this.currentText = this.targetText;
    this.textElement.textContent = this.targetText;
    this.charIndex = this.targetText.length;
    this.isTyping = false;
    this.skipButton.style.display = 'none';
    this.textCursor.style.display = 'none';

    if (this.onTypewriterComplete) {
      this.onTypewriterComplete();
      this.onTypewriterComplete = null;
    }
  }

  /**
   * Show player choices
   */
  showChoices(choices) {
    this.choicesElement.innerHTML = '';
    this.choicesElement.style.display = '';

    // Filter choices by conditions
    const availableChoices = choices.filter(choice => {
      if (!choice.condition) return true;
      return this.evaluateCondition(choice.condition);
    });

    if (availableChoices.length === 0) {
      // No valid choices, auto-advance or end
      this.advance();
      return;
    }

    // Create choice buttons
    availableChoices.forEach((choice, index) => {
      const button = document.createElement('button');
      button.className = 'dialog-choice';
      button.innerHTML = `
        <span class="choice-number">${index + 1}</span>
        <span class="choice-text">${choice.text}</span>
      `;

      button.addEventListener('click', () => {
        this.selectChoice(index);
      });

      // Hover sound
      button.addEventListener('mouseenter', () => {
        this.gameManager.audioManager.playSfx('menu_hover', { volume: 0.2 });
      });

      this.choicesElement.appendChild(button);
    });

    // Focus first choice
    const firstChoice = this.choicesElement.querySelector('.dialog-choice');
    if (firstChoice) {
      firstChoice.focus();
    }
  }

  /**
   * Player selected a choice
   */
  selectChoice(index) {
    const choices = this.currentNode.choices.filter(c => {
      if (!c.condition) return true;
      return this.evaluateCondition(c.condition);
    });

    const choice = choices[index];
    if (!choice) return;

    // Play confirm sound
    this.gameManager.audioManager.playSfx('dialog_select', { volume: 0.4 });

    // Apply consequences
    if (choice.consequences) {
      this.applyConsequences(choice.consequences);
    }

    // Store in history
    this.dialogHistory[this.dialogHistory.length - 1].choice = choice.text;

    // Emit choice event
    this.gameManager.emit('dialog:choice', {
      nodeId: this.currentNode.id,
      choice: choice
    });

    // Go to next node
    if (choice.nextNode) {
      this.goToNode(choice.nextNode);
    } else {
      this.endDialog();
    }
  }

  /**
   * Apply choice consequences
   */
  applyConsequences(consequences) {
    for (const consequence of consequences) {
      // Parse consequence string (e.g., "relationship:+1", "trust:-1", "flag:met_character")
      const [key, value] = consequence.split(':');

      if (value.startsWith('+') || value.startsWith('-')) {
        // Numeric value
        const numValue = parseInt(value);
        const current = this.flags[key] || 0;
        this.flags[key] = current + numValue;
      } else {
        // Flag set
        this.flags[key] = value;
      }
    }

    // Save flags
    this.saveFlags();
  }

  /**
   * Evaluate a condition
   */
  evaluateCondition(condition) {
    // Condition can be:
    // - "flag_name" - true if flag exists
    // - "!flag_name" - true if flag doesn't exist
    // - "flag_name:value" - true if flag equals value
    // - "flag_name>5" - numeric comparison

    if (condition.startsWith('!')) {
      const flag = condition.substring(1);
      return !this.flags[flag];
    }

    if (condition.includes('>')) {
      const [flag, value] = condition.split('>');
      return (this.flags[flag] || 0) > parseInt(value);
    }

    if (condition.includes('<')) {
      const [flag, value] = condition.split('<');
      return (this.flags[flag] || 0) < parseInt(value);
    }

    if (condition.includes(':')) {
      const [flag, value] = condition.split(':');
      return this.flags[flag] === value;
    }

    // Simple flag check
    return !!this.flags[condition];
  }

  /**
   * Advance to next node
   */
  advance() {
    if (!this.currentNode) return;

    if (this.currentNode.nextNode) {
      this.goToNode(this.currentNode.nextNode);
    } else {
      this.endDialog();
    }
  }

  /**
   * End the current dialog
   */
  endDialog() {
    this.isActive = false;
    this.currentDialog = null;
    this.currentNode = null;
    this.container.style.display = 'none';
    this.container.classList.remove('active');

    // Emit event with history
    this.gameManager.emit('dialog:ended', {
      history: this.dialogHistory,
      flags: this.flags
    });
  }

  /**
   * Load dialog data
   */
  loadDialog(dialogId) {
    // This would load from a file or data store
    // For now, return mock data structure
    return this.dialogData[dialogId];
  }

  /**
   * Get speaker info
   */
  getSpeakerInfo(speakerId) {
    return this.speakerData[speakerId] || null;
  }

  /**
   * Play voice line
   */
  playVoiceLine(voiceLine) {
    this.gameManager.audioManager.playSfx(voiceLine, {
      volume: this.config.voiceVolume
    });
  }

  /**
   * Save flags to persistent storage
   */
  saveFlags() {
    try {
      localStorage.setItem('dialog_flags', JSON.stringify(this.flags));
    } catch (e) {
      this.logger.warn('Failed to save dialog flags:', e);
    }
  }

  /**
   * Load flags from persistent storage
   */
  loadFlags() {
    try {
      const saved = localStorage.getItem('dialog_flags');
      if (saved) {
        this.flags = JSON.parse(saved);
      }
    } catch (e) {
      this.logger.warn('Failed to load dialog flags:', e);
    }
  }

  /**
   * Set a flag manually (for external systems)
   */
  setFlag(flag, value) {
    this.flags[flag] = value;
    this.saveFlags();
  }

  /**
   * Get a flag value
   */
  getFlag(flag) {
    return this.flags[flag];
  }

  /**
   * Check if a flag exists
   */
  hasFlag(flag) {
    return !!this.flags[flag];
  }

  // Sample dialog data
  dialogData = {
    'intro_conversation': {
      nodes: {
        'start': {
          speakerId: 'narrator',
          text: 'You find yourself standing before an ancient door, covered in mysterious symbols.',
          nextNode: 'door_inspect'
        },
        'door_inspect': {
          speakerId: 'player',
          text: 'What are these markings? They seem to... pulse with an inner light.',
          nextNode: 'voice_appears'
        },
        'voice_appears': {
          speakerId: 'mysterious_voice',
          text: 'At long last... another seeker approaches. Tell me, child of shadow, what do you seek?',
          choices: [
            {
              text: 'Answers. I need to know what happened here.',
              nextNode: 'response_answers',
              consequences: ['curiosity:+1']
            },
            {
              text: 'Who are you? Show yourself!',
              nextNode: 'response_challenge',
              consequences: ['aggression:+1']
            },
            {
              text: 'I... I\'m not sure. I was drawn here.',
              nextNode: 'response_uncertain',
              consequences: ['honesty:+1']
            },
            {
              text: '*Stay silent and observe*',
              nextNode: 'response_silent',
              consequences: ['caution:+1'],
              condition: 'caution>0'  // Only if player has been cautious before
            }
          ]
        },
        'response_answers': {
          speakerId: 'mysterious_voice',
          text: 'Answers... Yes. The truth you seek is heavy. Are you strong enough to carry it?',
          nextNode: 'choice_accept'
        },
        'response_challenge': {
          speakerId: 'mysterious_voice',
          text: 'Bold words. But bravery without wisdom leads only to an early grave.',
          nextNode: 'choice_accept'
        },
        'response_uncertain': {
          speakerId: 'mysterious_voice',
          text: 'Honesty. A rare quality in these shadowed times. Perhaps uncertainty is the beginning of wisdom.',
          nextNode: 'choice_accept'
        },
        'response_silent': {
          speakerId: 'mysterious_voice',
          text: 'Wise. You listen before speaking. The shadows respect those who take care.',
          nextNode: 'choice_accept'
        },
        'choice_accept': {
          speakerId: 'mysterious_voice',
          text: 'Very well. Step forward, and let us see what you are made of.',
          action: 'open_door',
          nextNode: null
        }
      }
    }
  };

  speakerData = {
    'narrator': {
      displayName: '',
      portrait: null
    },
    'player': {
      displayName: 'You',
      portrait: '/images/portraits/player.png'
    },
    'mysterious_voice': {
      displayName: '???',
      portrait: '/images/portraits/mysterious_voice.png'
    }
  };
}

export default DialogManager;
```

---

## 📝 How To Build Dialog Like This

### Step 1: Define Dialog Data Structure

```javascript
const simpleDialog = {
  nodes: {
    'greeting': {
      speakerId: 'npc',
      text: 'Hello there, traveler!',
      choices: [
        { text: 'Hi!', nextNode: 'friendly' },
        { text: 'Leave me alone.', nextNode: 'rude' }
      ]
    },
    'friendly': {
      speakerId: 'npc',
      text: 'What a pleasant surprise to meet a friendly face.',
      nextNode: null
    },
    'rude': {
      speakerId: 'npc',
      text: 'Well then. I\'ll be on my way.',
      nextNode: null
    }
  }
};
```

### Step 2: Create Basic Dialog Display

```javascript
function showDialog(dialog) {
  const container = document.createElement('div');
  container.className = 'dialog-overlay';
  container.innerHTML = `
    <div class="dialog-box">
      <p class="dialog-text"></p>
      <div class="choices"></div>
    </div>
  `;
  document.body.appendChild(container);

  let currentNode = dialog.nodes.greeting;

  function showNode(node) {
    container.querySelector('.dialog-text').textContent = node.text;

    const choicesDiv = container.querySelector('.choices');
    choicesDiv.innerHTML = '';

    if (node.choices) {
      node.choices.forEach(choice => {
        const btn = document.createElement('button');
        btn.textContent = choice.text;
        btn.onclick = () => {
          if (choice.nextNode) {
            showNode(dialog.nodes[choice.nextNode]);
          } else {
            container.remove();
          }
        };
        choicesDiv.appendChild(btn);
      });
    }
  }

  showNode(currentNode);
}
```

---

## 🔧 Advanced Dialog Features

### Conditional Choices

```javascript
// Only show choice if condition met
{
  text: 'I know about your secret.',
  nextNode: 'confront',
  condition: 'learned_secret'  // Flag must be set
}

// Numeric condition
{
  text: 'We meet again, old friend.',
  nextNode: 'friend_greeting',
  condition: 'friendship>5'  // Friendship level > 5
}

// Negative condition
{
  text: 'Wait, I remember you!',
  nextNode: 'recognition',
  condition: '!met_before'  // Haven't met yet
}
```

### Branching and Rejoining

```
     Start
       |
    ┌──┴──┐
   A     B   (Player makes choice)
    \   /
     \ /
      C   (Paths rejoin)
      |
    ┌──┴──┐
   D     E
```

### Consequences System

```javascript
// Immediate consequences
consequences: [
  'relationship:+1',      // Increase relationship
  'trust:-1',            // Decrease trust
  'met_character:true',  // Set flag
  'killed_dragon:false'  // Set boolean
]

// Check consequences later
{
  text: 'Thanks for saving my life!',
  condition: 'saved_village'
}
```

---

## Common Mistakes Beginners Make

### 1. Wall of Text

```javascript
// ❌ WRONG: Too much at once
{
  text: 'Welcome to the Kingdom of Eldoria! Founded in the Age of Myth...'
  // 500 words of lore
}
// Player skips it

// ✅ CORRECT: Break it up
{
  text: 'Welcome to the Kingdom of Eldoria.',
  nextNode: 'lore_continued'
}
// Digestible chunks
```

### 2. Illusion of Choice

```javascript
// ❌ WRONG: All choices same outcome
{
  choices: [
    { text: 'Yes', nextNode: 'same_result' },
    { text: 'No', nextNode: 'same_result' },
    { text: 'Maybe', nextNode: 'same_result' }
  ]
}
// Player feels tricked

// ✅ CORRECT: Meaningful differences
{
  choices: [
    {
      text: 'Yes',
      nextNode: 'accept_path',
      consequences: ['relationship:+1', 'quest_taken:true']
    },
    {
      text: 'No',
      nextNode: 'refuse_path',
      consequences: ['relationship:-1']
    }
  ]
}
// Player's choice matters
```

### 3. No Speaker Identification

```javascript
// ❌ WRONG: Who's talking?
{
  text: 'I think we should go this way.'
  // No speaker info
}
// Confusing

// ✅ CORRECT: Clear speaker
{
  speakerId: 'companion_rogue',
  text: 'I think we should go this way.',
  expression: 'thoughtful'
}
// Clear context
```

### 4. Too Fast Text

```javascript
// ❌ WRONG: Text appears instantly
showText(text) {
  element.textContent = text;
}
// Hard to read, no pacing

// ✅ CORRECT: Typewriter effect
showText(text) {
  let index = 0;
  const interval = setInterval(() => {
    element.textContent += text[index];
    index++;
    if (index >= text.length) clearInterval(interval);
  }, 30);
}
// Natural reading pace
```

---

## CSS for Dialog UI

```css
.dialog-container {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 2rem;
  pointer-events: none;
  z-index: 500;
}

.dialog-box {
  max-width: 800px;
  margin: 0 auto;
  background: rgba(10, 10, 26, 0.95);
  border: 2px solid rgba(0, 255, 136, 0.3);
  border-radius: 12px;
  padding: 1.5rem;
  pointer-events: auto;
  display: flex;
  gap: 1rem;
  box-shadow: 0 0 30px rgba(0, 0, 0, 0.5);
}

.dialog-portrait {
  width: 120px;
  flex-shrink: 0;
}

.portrait-image {
  width: 100%;
  height: 120px;
  object-fit: cover;
  border-radius: 8px;
  border: 2px solid rgba(255, 255, 255, 0.2);
}

.portrait-name {
  text-align: center;
  margin-top: 0.5rem;
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.8);
}

.dialog-content {
  flex: 1;
}

.dialog-speaker {
  font-size: 0.9rem;
  color: rgba(0, 255, 136, 0.8);
  margin-bottom: 0.5rem;
  font-weight: 500;
}

.dialog-text {
  font-size: 1.1rem;
  line-height: 1.6;
  color: white;
  min-height: 3rem;
}

.text-cursor {
  display: inline-block;
  animation: blink 0.8s infinite;
  color: rgba(0, 255, 136, 0.8);
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

.dialog-choices {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-top: 1rem;
}

.dialog-choice {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.75rem 1rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  color: white;
  cursor: pointer;
  transition: all 0.3s ease;
  text-align: left;
}

.dialog-choice:hover,
.dialog-choice:focus {
  background: rgba(0, 255, 136, 0.1);
  border-color: rgba(0, 255, 136, 0.4);
  transform: translateX(5px);
}

.choice-number {
  font-size: 0.8rem;
  color: rgba(0, 255, 136, 0.6);
  font-weight: bold;
}

.dialog-controls {
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
  margin-top: 1rem;
}

.skip-btn,
.continue-btn {
  padding: 0.5rem 1rem;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  color: white;
  cursor: pointer;
  font-size: 0.9rem;
}

.skip-btn:hover,
.continue-btn:hover {
  background: rgba(0, 255, 136, 0.2);
  border-color: rgba(0, 255, 136, 0.4);
}

/* Portrait expressions */
.portrait-image.expression-happy {
  border-color: rgba(255, 215, 0, 0.5);
}

.portrait-image.expression-angry {
  border-color: rgba(255, 50, 50, 0.5);
}

.portrait-image.expression-sad {
  border-color: rgba(100, 100, 200, 0.5);
}

.portrait-image.expression-neutral {
  border-color: rgba(255, 255, 255, 0.2);
}
```

---

## Related Systems

- [UIManager](./ui-manager.md) - Screen and overlay management
- [Save/Load System](../03-content/save-system.md) - Persisting dialog flags
- [Quest System](../03-content/quest-manager.md) - Dialog-driven quest updates

---

## Source File Reference

**Primary Files**:
- `../src/ui/DialogManager.js` - Dialog system (estimated)

**Key Classes**:
- `DialogManager` - Main dialog coordination
- Dialog data structure definitions

**Dependencies**:
- localStorage (flag persistence)
- AudioManager (voice lines, sfx)
- GameManager (event emission)

---

## References

- [Twine Documentation](https://twinery.org/cookbook/) - Interactive narrative concepts
- [Yarn Spinner](https://yarnspinner.dev/) - Dialog tools
- [Ink Scripting Language](https://www.inklestudios.com/ink/) - Narrative scripting

*Documentation last updated: January 12, 2026*
