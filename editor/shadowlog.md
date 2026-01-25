# 17 jan

i am at kammo's. I have reverse engineered the shadow of czar game and made an editor around this webgpu splats game. Now i can edit and debug and modify the game in this editor mode and learn how it works. I have a pret solid understanding of how the architecture works. We are thinking of the narrative as an experience. 

help me plan these enhancements by taking my proper input for how and what I need one by one for this sections, we will create a complete plan for all editor enhancements and make tickets.  

The Editor loads the first scene of the game the alley and phonebooth scene in the viewport. From the scene graph view, like the scenemanager view we should be able to jump to a specific scene and that scene and its files should be loaded in the editor. So in the scene manager ofcourse everything is not going to be a linear progression, dialogs, interaction points inside . A scene in our definition here is the zone - the stage. we need to be able to click them and land in those, and rearrange there sequence and add events to trigger the next zone etc. We should also be able to create a blank zone and add things to it, and then connect that node in our scene graph. Lets reimagine how this intuitive scengraph and story zone sequencer can be made.

- One zone can host many different game states.
- Game state is what happens

 Phase 1: Story First (The "What")

  1. Story Outline
     ├── Act 1: Setup (The Hook)
     ├── Act 2: Confrontation (The Journey)
     └── Act 3: Resolution (The Climax)

  2. Define Your Narrative States (GAME_STATES)
     ├── START_SCREEN
     ├── INTRO
     ├── FIRST_INTERACTION
     ├── CHOICE_POINT_1
     ├── ... 
     └── OUTRO

- zone is the stage, where it happens

 Zone Layout (Physical World)
     ├── Draw a map of connected spaces
     ├── Define zone boundaries
     └── Plan player movement paths