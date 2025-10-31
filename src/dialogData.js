/**
 * Dialog Data Structure
 *
 * Each dialog sequence contains:
 * - id: Unique identifier for the dialog
 * - audio: Path to the audio file
 * - preload: If true, load during loading screen; if false, load after (default: false)
 * - captions: Array of caption objects with:
 *   - text: The text to display
 *   - duration: How long to show this caption (in seconds)
 * - criteria: Optional object with key-value pairs that must match game state
 *   - Simple equality: { currentState: GAME_STATES.TITLE_SEQUENCE_COMPLETE }
 *   - Comparison operators: { currentState: { $gte: GAME_STATES.INTRO, $lt: GAME_STATES.DRIVE_BY } }
 *   - Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
 *   - Example: Play after INTRO but before DRIVE_BY
 * - once: If true, only play once (tracked automatically)
 * - priority: Higher priority dialogs are checked first (default: 0)
 * - autoPlay: If true, automatically play when conditions are met (default: false)
 * - delay: Delay in seconds before playing after state conditions are met (default: 0)
 * - playNext: Chain to another dialog after this one completes
 *   - Can be a dialog object (e.g., dialogTracks.nextDialog) or string ID (e.g., "nextDialog")
 *   - Allows creating dialog sequences without requiring game state changes between lines
 *   - The chained dialog's delay property is respected if specified
 *   - Example: playNext: dialogTracks.continueConversation
 * - onComplete: Optional function called when dialog completes, receives gameManager
 *   - Example: (gameManager) => gameManager.setState({ currentState: GAME_STATES.NEXT_STATE })
 *   - Note: When using playNext, onComplete is called before moving to the next dialog
 *
 * Note: For multiple choice dialogs, see dialogChoiceData.js
 *
 * Usage:
 * import { dialogTracks } from './dialogData.js';
 * dialogManager.playDialog(dialogTracks.intro);
 * // or in dialogChoiceData.js:
 * triggerDialog: dialogTracks.bonneSoiree
 */

import { GAME_STATES, DIALOG_RESPONSE_TYPES } from "./gameData.js";
import { checkCriteria } from "./utils/criteriaHelper.js";

export const dialogTracks = {
  intro: {
    id: "intro",
    audio: "./audio/dialog/cole-on-her-trail.mp3",
    preload: true,
    captions: [
      { text: "I'd been on her trail for weeks.", duration: 2.0 },
      { text: "An art thief, she'd swindled society-types,", duration: 3.5 },
      {
        text: "hauling in more than a few of the Old Masters.",
        duration: 2.5,
      },
      {
        text: "An anonymous tip came in:",
        duration: 2.5,
      },
      {
        text: "the stash was uptown,",
        duration: 2.0,
      },
      {
        text: "and sure as I staked it out, she was there.",
        duration: 2.0,
      },
      {
        text: "Time to answer some tough questions, Ms. LeClaire.",
        duration: 2.5,
      },
    ],
    criteria: { currentState: GAME_STATES.INTRO },
    once: true,
    autoPlay: true,
    priority: 100,
    delay: 1.0,
    progressStateTrigger: { progress: 0.85, state: GAME_STATES.TITLE_SEQUENCE },
  },

  // Radio captions (audio is handled by sfxData.js with reactive light)
  radioCaptions: {
    id: "radioCaptions",
    // No audio - the radio SFX plays independently with reactive light
    captions: [
      { text: '[Duke Ellington\'s "The Mooche" plays]', duration: 2.75 },
      { text: "The Czar strikes again!", duration: 1.5 },
      { text: "Brazen brute bashes bank!", duration: 2.0 },
      { text: "Czar's zealots embezzle zillions!", duration: 2.25 },
      { text: "Cops can't quell criminal caper!", duration: 2.0 },
      { text: "This and more tonight, on City Beat!", duration: 2.25 },
    ],
    criteria: { currentState: GAME_STATES.NEAR_RADIO },
    once: true,
    autoPlay: true,
    priority: 100,
    delay: 0,
  },

  heyYouBeingWatched: {
    id: "heyYouBeingWatched",
    audio: "./audio/dialog/cole-hey-you-being-watched.mp3",
    preload: false, // Load after loading screen
    captions: [
      { text: "Hey, you!", duration: 2.0 },
      { text: "Feels like I'm being watched...", duration: 3.5 },
    ],
    criteria: { shadowGlimpse: true },
    once: true,
    autoPlay: true,
    priority: 100,
    delay: 3.0,
  },

  // Dialog that plays when phone starts ringing
  okayICanTakeAHint: {
    id: "okayICanTakeAHint",
    audio: "./audio/dialog/cole-okay-i-can-take-a-hint.mp3",
    preload: false, // Load after loading screen
    captions: [{ text: "Okay, I can take a hint.", duration: 2.0 }],
    criteria: { currentState: GAME_STATES.PHONE_BOOTH_RINGING },
    once: true,
    autoPlay: true,
    priority: 100,
    delay: 5,
  },

  // Dialog that triggers first choice moment
  bonsoir: {
    id: "bonsoir",
    audio: "./audio/dialog/leclaire-bonsoir.mp3",
    preload: false, // Load after loading screen
    captions: [
      { text: "Bonsoir...", duration: 1.5 },
      { text: "I presume you know who this is?", duration: 2 },
    ],
    criteria: { currentState: GAME_STATES.ANSWERED_PHONE },
    once: true,
    autoPlay: true,
    priority: 100,
    delay: 2.0, // Wait 2 seconds after answering phone
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.DIALOG_CHOICE_1 });
    },
  },

  // Follow-up dialog for EMPATH response
  dialogChoice1Empath: {
    id: "dialogChoice1Empath",
    audio: "./audio/dialog/choice-1_empath_someone-who-made-a-mistake.mp3",
    preload: false, // Load after loading screen
    captions: [
      { text: "Someone who made a little mistake, that's all.", duration: 2.5 },
    ],
    criteria: {
      currentState: GAME_STATES.DIALOG_CHOICE_1,
      dialogChoice1: DIALOG_RESPONSE_TYPES.EMPATH,
    },
    once: true,
    autoPlay: true,
    priority: 100,
    delay: 1.0,
    onComplete: (gameManager) => {
      gameManager.setState({ dialogChoice1Response: true });
    },
  },

  // Follow-up dialog for PSYCHOLOGIST response
  dialogChoice1Psychologist: {
    id: "dialogChoice1Psychologist",
    audio:
      "./audio/dialog/choice-1_psych_someone-who-was-never-taught-better.mp3",
    captions: [
      {
        text: "Someone who was never taught better than the ways of a thief.",
        duration: 2.5,
      },
    ],
    criteria: {
      currentState: GAME_STATES.DIALOG_CHOICE_1,
      dialogChoice1: DIALOG_RESPONSE_TYPES.PSYCHOLOGIST,
    },
    once: true,
    autoPlay: true,
    priority: 100,
    delay: 1.0,
    onComplete: (gameManager) => {
      gameManager.setState({ dialogChoice1Response: true });
    },
  },

  dialogChoice1Lawful: {
    id: "dialogChoice1Lawful",
    audio: "./audio/dialog/choice-1_lawful_someone-with-stolen-property.mp3",
    preload: false,
    captions: [
      {
        text: "Someone with stolen property in their possession",
        duration: 2.0,
      },
      { text: "who might be in a lot of trouble!", duration: 2.0 },
    ],
    criteria: {
      currentState: GAME_STATES.DIALOG_CHOICE_1,
      dialogChoice1: DIALOG_RESPONSE_TYPES.LAWFUL,
    },
    once: true,
    autoPlay: true,
    priority: 100,
    delay: 1.0,
    onComplete: (gameManager) => {
      gameManager.setState({ dialogChoice1Response: true });
    },
  },

  // PETIT's responses to player's dialog choices
  dialogChoice1EmpathResponse: {
    id: "dialogChoice1EmpathResponse",
    audio: "./audio/dialog/resp-1_empath_oui-and-ive-made-so-many.mp3",
    preload: false, // Load after loading screen
    captions: [{ text: "Oui, and I've made *so* many.", duration: 2.5 }],
    criteria: {
      currentState: GAME_STATES.DIALOG_CHOICE_1,
      dialogChoice1: DIALOG_RESPONSE_TYPES.EMPATH,
      dialogChoice1Response: true,
    },
    once: true,
    autoPlay: true,
    priority: 99,
    delay: 0.5,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.DRIVE_BY_PREAMBLE });
    },
  },

  dialogChoice1PsychologistResponse: {
    id: "dialogChoice1PsychologistResponse",
    audio: "./audio/dialog/resp-1_psych_im-sure-youll-educate-me.mp3",
    preload: false, // Load after loading screen
    captions: [{ text: "I'm sure you will educate me...", duration: 2.0 }],
    criteria: {
      currentState: GAME_STATES.DIALOG_CHOICE_1,
      dialogChoice1: DIALOG_RESPONSE_TYPES.PSYCHOLOGIST,
      dialogChoice1Response: true,
    },
    once: true,
    autoPlay: true,
    priority: 99,
    delay: 0.5,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.DRIVE_BY_PREAMBLE });
    },
  },

  dialogChoice1LawfulResponse: {
    id: "dialogChoice1LawfulResponse",
    audio: "./audio/dialog/resp-1_lawful_hm-quite-the-lawman-you-are.mp3",
    preload: false, // Load after loading screen
    captions: [{ text: "Hm, quite the lawman you are.", duration: 2.0 }],
    criteria: {
      currentState: GAME_STATES.DIALOG_CHOICE_1,
      dialogChoice1: DIALOG_RESPONSE_TYPES.LAWFUL,
      dialogChoice1Response: true,
    },
    once: true,
    autoPlay: true,
    priority: 99,
    delay: 0.5,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.DRIVE_BY_PREAMBLE });
    },
  },

  // Cole's response to choosing "Good kitty" for the cat
  coleGoodKitty: {
    id: "coleGoodKitty",
    audio: "./audio/dialog/cole-aw-good-kitty.mp3",
    preload: false, // Load after loading screen
    captions: [{ text: "Aw, good kitty.", duration: 2.0 }],
    criteria: {
      currentState: GAME_STATES.CAT_DIALOG_CHOICE,
      catDialogChoice: DIALOG_RESPONSE_TYPES.CAT_GOOD_KITTY,
    },
    once: true,
    autoPlay: true,
    priority: 100,
    delay: 0.5,
  },

  // Cole's response to choosing "Damn cats" for the cat
  coleDamnCats: {
    id: "coleDamnCats",
    audio: "./audio/dialog/cole-damn-cats.mp3",
    preload: false, // Load after loading screen
    captions: [{ text: "Damn cats.", duration: 2.0 }],
    criteria: {
      currentState: GAME_STATES.CAT_DIALOG_CHOICE,
      catDialogChoice: DIALOG_RESPONSE_TYPES.CAT_DAMN_CATS,
    },
    once: true,
    autoPlay: true,
    priority: 100,
    delay: 0.5,
  },

  // Warning dialog after PETIT's response
  theyreHereForYou: {
    id: "theyreHereForYou",
    audio: "./audio/dialog/04-theyre-here-for-you-duck-and-cover-now.mp3",
    preload: false, // Load after loading screen
    captions: [
      { text: "They're here for you!", duration: 1.65 },
      { text: "Duck and cover, now!", duration: 2.0 },
    ],
    criteria: {
      currentState: GAME_STATES.DRIVE_BY_PREAMBLE,
    },
    once: true,
    autoPlay: true,
    priority: 98,
    delay: 2.125,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.DRIVE_BY });
    },
  },

  // PETIT's callout after the drive-by-shooting
  postDriveBy: {
    id: "postDriveBy",
    audio: "./audio/dialog/leclaire-so-did-you-survive.mp3",
    captions: [
      { text: "So...", duration: 1.0 },
      { text: "Did you survive?", duration: 1.75 },
      { text: "Then hightail it!", duration: 1.5 },
      {
        text: "There's a room nearby.",
        duration: 1.5,
      },
      { text: "I've something to show you.", duration: 2.0 },
    ],
    criteria: {
      currentState: GAME_STATES.POST_DRIVE_BY,
    },
    once: true,
    autoPlay: true,
    priority: 100,
    delay: 0.5,
  },

  // PETIT's final warning before drive-by
  itsLeCzar: {
    id: "itsLeCzar",
    audio: "./audio/dialog/leclaire-its-le-czar.mp3",
    captions: [
      { text: "Cole,", duration: 1.0 },
      { text: "it's Le Czar...", duration: 1.75 },
      { text: "He's gone bad!", duration: 1.0 },
      {
        text: "He was always bad!",
        duration: 1.55,
      },
      { text: "Non, c'est différent...", duration: 1.75 },
      { text: "Under the spell of a mad cultist.", duration: 3.75 },
      { text: "And that thing...", duration: 1.5 },
    ],
    criteria: {
      currentState: GAME_STATES.OFFICE_PHONE_ANSWERED,
    },
    once: true,
    autoPlay: true,
    priority: 100,
    delay: 0.5,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.PRE_VIEWMASTER });
    },
  },

  preViewmaster: {
    id: "preViewmaster",
    audio: "./audio/dialog/leclaire-une-technologie.mp3",
    captions: [
      { text: "Une technologie incroyable...", duration: 2.0 },
      { text: "It can show you many things.", duration: 2.0 },
    ],
    criteria: { currentState: GAME_STATES.PRE_VIEWMASTER },
    once: true,
    autoPlay: true,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.VIEWMASTER });
    },
  },

  butIDont: {
    id: "butIDont",
    audio: "./audio/dialog/cole-but-i-dont.mp3",
    captions: [
      { text: "But I don't...", duration: 3.0 },
      { text: "Ah...", duration: 2.75 },
      { text: "Gee, this... this is...", duration: 2.75 },
      { text: "This is really something!", duration: 3.0 },
    ],
    criteria: { currentState: GAME_STATES.VIEWMASTER },
    once: true,
    autoPlay: true,
    delay: 0.5,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.VIEWMASTER_COLOR });
    },
  },

  itIsBeautiful: {
    id: "itIsBeautiful",
    audio: "./audio/dialog/leclaire-it-is-beautiful.mp3",
    captions: [
      { text: "It is beautiful, non?", duration: 2.0 },
      { text: "And yet they can control your perception.", duration: 3.25 },
      { text: "Observe.", duration: 1.5 },
    ],
    criteria: { currentState: GAME_STATES.VIEWMASTER_COLOR },
    once: true,
    autoPlay: true,
    delay: 0.5,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.VIEWMASTER_DISSOLVE });
    },
  },

  whatTheHeck: {
    id: "whatTheHeck",
    audio: "./audio/dialog/cole-what-the-heck.mp3",
    captions: [{ text: "What the heck?!", duration: 1.5 }],
    criteria: { currentState: GAME_STATES.VIEWMASTER_DISSOLVE },
    once: true,
    autoPlay: true,
    delay: 4.5,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.VIEWMASTER_DIALOG });
    },
  },

  thatWasNothing: {
    id: "thatWasNothing",
    audio: "./audio/dialog/leclaire-that-was-nothing.mp3",
    captions: [{ text: "That was nothing...", duration: 1.5 }],
    criteria: { currentState: GAME_STATES.VIEWMASTER_DIALOG },
    once: true,
    autoPlay: true,
    delay: 1.0,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.VIEWMASTER_HELL });
    },
  },

  hachiMachi: {
    id: "hachiMachi",
    audio: "./audio/dialog/cole-hachi-machi.mp3",
    captions: [
      { text: "Hachi machi...", duration: 2.5 },
      { text: "", duration: 0.5 },
      { text: "Is this real?", duration: 2.5 },
    ],
    criteria: { currentState: GAME_STATES.VIEWMASTER_HELL },
    once: true,
    autoPlay: true,
    delay: 4.0,
    playNext: "remainHereTooLong",
  },

  remainHereTooLong: {
    id: "remainHereTooLong",
    audio: "./audio/dialog/leclaire-is-anything.mp3",
    captions: [
      { text: "Is anything?", duration: 1.5 },
      { text: "They can make of this world a hellish place.", duration: 3.5 },
      { text: "Remain here too long", duration: 2.5 },
      { text: "And you'll see the world only as they wish it.", duration: 3.5 },
      { text: "Mind control!", duration: 1.75 },
      { text: "Exactement!", duration: 1.75 },
    ],
    once: true,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.POST_VIEWMASTER });
    },
  },

  cat2DialogFriend: {
    id: "cat2DialogFriend",
    audio: "./audio/dialog/cole-theres-my-friend.mp3",
    captions: [{ text: "Hey, there's my friend.", duration: 2.3 }],
    criteria: {
      currentState: GAME_STATES.CAT_DIALOG_CHOICE_2,
      catDialogChoice2: DIALOG_RESPONSE_TYPES.CAT_MY_FRIEND,
    },
    once: true,
    autoPlay: true,
    delay: 0.5,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.PRE_EDISON });
    },
  },

  cat2DialogGit: {
    id: "cat2DialogGit",
    audio: "./audio/dialog/cole-you-again-git.mp3",
    captions: [{ text: "You again? Git!", duration: 2.3 }],
    criteria: {
      currentState: GAME_STATES.CAT_DIALOG_CHOICE_2,
      catDialogChoice2: DIALOG_RESPONSE_TYPES.CAT_GIT,
    },
    once: true,
    autoPlay: true,
    delay: 0.5,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.PRE_EDISON });
    },
  },

  whatColeFocus: {
    id: "whatColeFocus",
    audio: "./audio/dialog/leclaire-what-cole-focus.mp3",
    captions: [
      { text: "What?", duration: 1.5 },
      { text: "Cole, focus!", duration: 1.5 },
      { text: "This time we've got him - on the record!", duration: 3.0 },
      { text: "Check the Edison...", duration: 1.5 },
    ],
    once: true,
    autoPlay: true,
    criteria: { currentState: GAME_STATES.PRE_EDISON },
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.EDISON });
    },
  },

  iGetMyCorners: {
    id: "iGetMyCorners",
    audio: "./audio/dialog/czar-i-get-my-corners.mp3",
    captions: [
      { text: "I get my corners,", duration: 2.0 },
      { text: "the drug trade,", duration: 2.0 },
      { text: "the speakeasies,", duration: 2.0 },
      { text: "and you get...", duration: 2.5 },
      { text: "what you want, eh? [laughs]", duration: 2.0 },
      { text: "", duration: 1.5 },
      { text: "Quite.", duration: 1.5 },
    ],
    once: true,
    autoPlay: true,
    delay: 5.0,
    criteria: { currentState: GAME_STATES.EDISON },
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.DIALOG_CHOICE_2 });
    },
  },

  dialogChoice2Empath: {
    id: "dialogChoice2Empath",
    audio: "./audio/dialog/choice-2_empath_caught-red-handed.mp3",
    preload: false,
    captions: [
      { text: "Why, we've caught the Czar red-handed!", duration: 2.5 },
    ],
    criteria: {
      currentState: GAME_STATES.DIALOG_CHOICE_2,
      dialogChoice2: DIALOG_RESPONSE_TYPES.EMPATH,
    },
    once: true,
    autoPlay: true,
    priority: 100,
    delay: 0.5,
    playNext: "dialogChoice2EmpathResponse",
    onComplete: (gameManager) => {
      gameManager.setState({ dialogChoice2Response: true });
    },
  },

  dialogChoice2Psychologist: {
    id: "dialogChoice2Psychologist",
    audio: "./audio/dialog/choice-2_psych_cant-say-who.mp3",
    preload: false,
    captions: [{ text: "I can't say who's responsible yet.", duration: 2.5 }],
    criteria: {
      currentState: GAME_STATES.DIALOG_CHOICE_2,
      dialogChoice2: DIALOG_RESPONSE_TYPES.PSYCHOLOGIST,
    },
    once: true,
    autoPlay: true,
    priority: 100,
    delay: 0.5,
    playNext: "dialogChoice2PsychologistResponse",
    onComplete: (gameManager) => {
      gameManager.setState({ dialogChoice2Response: true });
    },
  },

  dialogChoice2Lawful: {
    id: "dialogChoice2Lawful",
    audio: "./audio/dialog/choice-2_lawful_some-ruse.mp3",
    preload: false,
    captions: [{ text: "How do I know this isn't some ruse?", duration: 2.5 }],
    criteria: {
      currentState: GAME_STATES.DIALOG_CHOICE_2,
      dialogChoice2: DIALOG_RESPONSE_TYPES.LAWFUL,
    },
    once: true,
    autoPlay: true,
    priority: 100,
    delay: 0.5,
    playNext: "dialogChoice2LawfulResponse",
    onComplete: (gameManager) => {
      gameManager.setState({ dialogChoice2Response: true });
    },
  },

  dialogChoice2EmpathResponse: {
    id: "dialogChoice2EmpathResponse",
    audio: "./audio/dialog/resp-2_empath_merci-cole.mp3",
    preload: false,
    captions: [
      { text: "Merci, Cole!", duration: 1.5 },
      { text: "I knew you would help!", duration: 2.0 },
    ],
    once: true,
    delay: 0,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.CZAR_STRUGGLE });
    },
  },

  dialogChoice2PsychologistResponse: {
    id: "dialogChoice2PsychologistResponse",
    audio: "./audio/dialog/resp-2_psych_youre-kidding.mp3",
    preload: false,
    captions: [{ text: "You're kidding... It's so obvious!", duration: 3.0 }],
    once: true,
    delay: 0,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.CZAR_STRUGGLE });
    },
  },

  dialogChoice2LawfulResponse: {
    id: "dialogChoice2LawfulResponse",
    audio: "./audio/dialog/resp-2_lawful_those-goons.mp3",
    preload: false,
    captions: [
      {
        text: "Those goons were going to hang this on you, dummy!",
        duration: 2.0,
      },
    ],
    once: true,
    delay: 0,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.CZAR_STRUGGLE });
    },
  },

  bravoDetective: {
    id: "bravoDetective",
    audio: "./audio/dialog/czar-bravo-detective.mp3",
    captions: [
      { text: "[LeClaire screams]", duration: 3.0 },
      { text: "[A commotion]", duration: 3.5 },
      { text: "Bravo, detective!", duration: 2.5 },
      { text: "Nothing gets by you.", duration: 3.0 },
      { text: "This... is the Czar.", duration: 4.0 },
    ],
    criteria: { currentState: GAME_STATES.CZAR_STRUGGLE },
    once: true,
    autoPlay: true,
    priority: 100,
    playNext: "twoBitCrook",
  },

  twoBitCrook: {
    id: "twoBitCrook",
    audio: "./audio/dialog/cole-you-mean-the-two-bit-crook.mp3",
    captions: [
      {
        text: "You mean the two-bit crook what fanicies himself a king?",
        duration: 4.0,
      },
      {
        text: "The one way in over his head with cultists and madmen?",
        duration: 3.0,
      },
      { text: "That Czar?", duration: 2.0 },
    ],
    once: true,
    priority: 100,
    playNext: "theVerySame",
  },

  theVerySame: {
    id: "theVerySame",
    audio: "./audio/dialog/czar-the-very-same.mp3",
    captions: [
      { text: "The very same, Cole.", duration: 2.0 },
      { text: "Now meet my new friend.", duration: 2.0 },
      { text: "Oh mister Shadow?", duration: 2.0 },
    ],
    once: true,
    priority: 100,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.SHOULDER_TAP });
    },
  },

  interuptHisRitual: {
    id: "interuptHisRitual",
    audio: "./audio/dialog/leclaire-you-have-to-interrupt-his-ritual.mp3",
    captions: [
      { text: "Go, you have to interrupt his ritual!", duration: 2.25 },
      { text: "Dark runes manifest throughout the hall!", duration: 2.5 },
      { text: "Use the View-Master to find them,", duration: 1.75 },
      {
        text: "but wear it too long and you'll lose your mind!",
        duration: 2.75,
      },
    ],
    criteria: { currentState: GAME_STATES.CURSOR },
    autoPlay: true,
    once: true,
    priority: 100,
    delay: 1.5,
  },

  // Drawing game failures (cycle through 3 responses)
  drawingFailure1: {
    id: "drawingFailure1",
    audio: "./audio/dialog/leclaire-drawing-failure-1.mp3",
    captions: [{ text: "That was wrong!", duration: 1.5 }],
    criteria: {
      currentState: GAME_STATES.CURSOR,
      lastDrawingSuccess: false,
      drawingFailureCount: 1,
    },
    autoPlay: true,
    once: false,
    priority: 90,
  },

  drawingFailure2: {
    id: "drawingFailure2",
    audio: "./audio/dialog/leclaire-drawing-failure-2.mp3",
    captions: [{ text: "Quoi? Try again.", duration: 1.0 }],
    criteria: {
      currentState: GAME_STATES.CURSOR,
      lastDrawingSuccess: false,
      drawingFailureCount: 2,
    },
    autoPlay: true,
    once: false,
    priority: 90,
  },

  drawingFailure3: {
    id: "drawingFailure3",
    audio: "./audio/dialog/leclaire-drawing-failure-3.mp3",
    captions: [{ text: "You must look for the symbol!", duration: 1.8 }],
    criteria: {
      currentState: GAME_STATES.CURSOR,
      lastDrawingSuccess: false,
      drawingFailureCount: 3,
    },
    autoPlay: true,
    once: false,
    priority: 90,
  },

  // Drawing game successes
  drawingSuccess1: {
    id: "drawingSuccess1",
    audio: "./audio/dialog/leclaire-drawing-success-1.mp3",
    captions: [{ text: "It worked! Now find the next.", duration: 2.75 }],
    criteria: {
      currentState: GAME_STATES.CURSOR,
      lastDrawingSuccess: true,
      drawingSuccessCount: 1,
    },
    autoPlay: true,
    once: true,
    priority: 100,
  },

  drawingSuccess2: {
    id: "drawingSuccess2",
    audio: "./audio/dialog/leclaire-drawing-success-2.mp3",
    captions: [{ text: "Yes! Just one more.", duration: 2.0 }],
    criteria: {
      currentState: GAME_STATES.CURSOR,
      lastDrawingSuccess: true,
      drawingSuccessCount: 2,
    },
    autoPlay: true,
    once: true,
    priority: 100,
  },
};

/**
 * Get dialog sequences that should play for the current game state
 * @param {Object} gameState - Current game state
 * @param {Set} playedDialogs - Set of dialog IDs that have already been played
 * @returns {Array} Array of dialog sequences that match conditions
 */
export function getDialogsForState(gameState, playedDialogs = new Set()) {
  // Convert to array and filter for autoPlay dialogs only
  const autoPlayDialogs = Object.values(dialogTracks).filter(
    (dialog) => dialog.autoPlay === true
  );

  // Sort by priority (descending)
  const sortedDialogs = autoPlayDialogs.sort(
    (a, b) => (b.priority || 0) - (a.priority || 0)
  );

  const matchingDialogs = [];

  for (const dialog of sortedDialogs) {
    // Skip if already played and marked as "once"
    if (dialog.once && playedDialogs.has(dialog.id)) {
      continue;
    }

    // Check criteria (supports operators like $gte, $lt, etc.)
    if (dialog.criteria) {
      if (!checkCriteria(gameState, dialog.criteria)) {
        continue;
      }
    }

    // If we get here, all conditions passed
    matchingDialogs.push(dialog);
  }

  return matchingDialogs;
}

export default dialogTracks;

// Example: Using playNext to chain dialogs without game state changes
//
// petitDialogPart1: {
//   id: "petitDialogPart1",
//   audio: "./audio/dialog/petit-part1.mp3",
//   captions: [{ text: "I didn't paint those paintings!", duration: 2.0 }],
//   criteria: { currentState: GAME_STATES.SOME_STATE },
//   once: true,
//   autoPlay: true,
//   playNext: "petitDialogPart2", // String ID reference
// },
//
// petitDialogPart2: {
//   id: "petitDialogPart2",
//   audio: "./audio/dialog/petit-part2.mp3",
//   captions: [{ text: "And I just saved your life!", duration: 3.5 }],
//   playNext: "petitDialogPart3", // Chain multiple dialogs
// },
//
// petitDialogPart3: {
//   id: "petitDialogPart3",
//   audio: "./audio/dialog/petit-part3.mp3",
//   captions: [{ text: "Those goons were going to hang this on you, dummy!", duration: 2.5 }],
//   onComplete: (gameManager) => {
//     gameManager.setState({ currentState: GAME_STATES.NEXT_STATE });
//   },
// },
