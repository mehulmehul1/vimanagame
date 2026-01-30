# 🎵 VIMANA – HARP DUET MINIGAME DESIGN DOCUMENT

## 🌌 OVERVIEW

This minigame is a **calm, musical call-and-response duet** between:

- **The Vimana (a living ship)** → initiates the musical phrase  
- **The Player** → responds using a harp

It takes place inside a **large, organic music chamber** within the ship. The mood must be **peaceful, generous, and meditative**. This is not meant to challenge reflexes or timing — it is about **listening, remembering, and responding musically**.

The goal of the minigame is to gradually form a **vortex** in the distance. The vortex represents the **harmony formed between player and ship**.

This minigame is:
- Easy
- Forgiving
- Non-stressful
- Musical and atmospheric

It is **not** a rhythm-timing challenge.

---

## 🌊 ENVIRONMENT CONTEXT

- The player stands on a **fixed platform** inside a vast water-filled chamber
- A **harp-like instrument** stands in front of the player
- The chamber opens toward a distant area where a **vortex forms**
- Jellyfish live in the water — they are extensions of the Vimana

The space should feel:
- Spacious
- Resonant
- Alive
- Safe

---

## 🎬 PHASE 1 — STARTING THE MINIGAME

**Trigger:** Player presses **E** near the harp

### System Changes
- Player movement is disabled
- Camera locks forward toward:
  - Harp (bottom third of screen)
  - Water surface (middle)
  - Vortex area (distance)

This begins **Performance Mode**.

The player is now in a **musical interaction state**, not normal gameplay.

---

## 🪼 PHASE 2 — THE VIMANA PLAYS (JELLYFISH PHRASE)

The Vimana begins the duet by presenting a **melodic sequence** using jellyfish.

### Jellyfish Behavior Per Note

For each note in the sequence:

1. A jellyfish jumps out of the water  
2. It emits a **soft musical tone**  
3. Its **vertical position is perpendicular to the harp string below and visually tell the player which strings to play**
4. After its tone finishes, it remains airborne until the phrase ends

### What Jellyfish DO NOT Communicate

Jellyfish do **NOT** encode:
- Rhythm accuracy  
- Timing precision  
- Note duration  
- Note type through glow or height  

This is **not a rhythm game**.

---

## 🎼 WHAT THE JELLYFISH ACTUALLY TEACH

They teach only one thing:

> **The ORDER of notes the player must play**

Each jellyfish corresponds to a harp string based on **horizontal alignment**.

⚠️ **IMPORTANT:**  
The table below is **only an example to explain the mechanic**.  
It does **not** represent a fixed layout or constant sequence in the game.

| Jellyfish Horizontal Position | Harp String | Meaning (Example Only) |
|-------------------------------|-------------|-------------------------|
| Left                          | Left string | Example Note A |
| Middle                        | Middle string | Example Note B |
| Right                         | Right string | Example Note C |

In the actual minigame:
- The number of possible strings may vary
- Positions may vary
- Sequences change each round

The core rule remains:

> **Jellyfish position shows WHICH string to play.  
> The ORDER they appear shows the SEQUENCE to remember.**

The sequence of jellyfish jumps = the note sequence.

---

## 🌊 SYNCHRONIZED LANDING (TURN SIGNAL)

After the final jellyfish tone ends:

- **All jellyfish fall back into the water at the SAME TIME**
- A **single unified splash** occurs

### Purpose
1. Prevents splash sounds from interfering with the melody
2. Clearly signals:

> 🎵 “The Vimana’s phrase is complete. Your turn.”

The splash is **not part of the sequence**.

---

## 🎹 PHASE 3 — PLAYER RESPONSE

The player now plays the harp.

### What the Player Must Do

The player must reproduce:

✔ The **same note sequence in the same order**

### What Does NOT Matter

❌ Timing precision  
❌ Speed of playing  
❌ Holding notes for exact durations  
❌ Pausing between notes  

This minigame tests **memory only**, not reflexes or rhythm skill.

Players are allowed to take their time.

---

## ✅ PHASE 4 — SUCCESS

If the player plays the correct sequence:

- A **segment of the vortex forms**
- The vortex becomes more stable
- A warm harmonic resonance plays
- The chamber subtly responds

This represents the duet achieving harmony.

---

## ❌ PHASE 4 — FAILURE

If the player plays the wrong sequence:

- A formed vortex piece **breaks apart**
- A soft disharmonic tone plays
- Jellyfish perform the phrase again

There is **no punishment**, only a gentle retry.

---

## 🔁 PHASE 5 — LOOP STRUCTURE

The duet continues in cycles:

**Ship plays → Player responds → World reacts**

The player must succeed **3–4 times**.

Each success builds another vortex segment.

When complete, the vortex stabilizes and progression continues.

---

## 🎭 MOOD & DESIGN PHILOSOPHY

This minigame must feel:

- Calm  
- Musical  
- Generous  
- Meditative  
- Like a duet, not a challenge

Failure should feel like:
> “We were slightly out of sync. Let’s try again.”

Success should feel like:
> “We are learning to sing together.”

---

## 🚫 STRICT DO NOT RULES

Do NOT design this as:
- A rhythm timing game  
- A reflex challenge  
- A precision input system  
- A punishment loop  

Do NOT require players to:
- Match tempo  
- Match note length  
- React quickly  

---

## 🎶 FINAL SUMMARY FOR AI SYSTEMS

**Type:** Musical memory duet  
**Core Mechanic:** Observe note order → repeat note order  
**Skill Tested:** Sequence memory only  
**Timing Skill Required:** None  
**Tone:** Calm, forgiving, atmospheric  
**Failure Cost:** None  
**Progression:** 3–4 correct sequences build vortex  

🪼 Jellyfish = Voice of the Vimana  
🎼 Harp = Voice of the Player  
🌀 Vortex = Harmony formed when both are in sync
