# PLACEHOLDER - Add your intro video here

## Video Requirements:

- **Format**: WebM with VP8/VP9 codec
- **Alpha Channel**: Supported for transparency (yuva420p pixel format)
- **Recommended Resolution**: 1920x1080
- **File**: intro.webm

## Creating WebM with Alpha using FFmpeg:

```bash
ffmpeg -i input.mov -c:v libvpx-vp9 -vf gloop,scale=1920:1080 \
  -auto-alt-ref 1 -pix_fmt yuva420p intro.webm
```

## What happens when video ends:

The video's `onComplete` callback will automatically:
1. Set game state to MUSIC_ROOM
2. Enable player controls
3. Teleport player to music room spawn point
