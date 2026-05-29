# forge-soniqo

Audio decomposition for Plato agents. Part of the **forge-flux** ecosystem.

## Concept

Plato agents process content as **tiles**. `forge-soniqo` is the audio-specific decomposer — it takes raw audio samples and splits them into temporal chunks, each annotated with spectral features.

No external audio libraries required. All features are computed from raw samples.

## Features

- **Chunking** — split audio into fixed-duration tiles (e.g., 100ms chunks)
- **Spectral features** — peak frequency, RMS energy, zero-crossing rate per chunk
- **WAV parsing** — built-in 16-bit PCM WAV decoder
- **Reassembly** — merge tiles back into a continuous sample buffer
- **Zero dependencies** — pure Rust, no audio frameworks

## Usage

```rust
use forge_soniqo::{AudioDecomposer, AudioTile};

let decomposer = AudioDecomposer::new(44100, 100); // 44.1kHz, 100ms chunks
let result = decomposer.decompose(&samples);

for tile in &result.tiles {
    println!("{}ms: peak={:.1}Hz rms={:.3} zcr={:.3}",
        tile.timestamp_ms, tile.spectral_peak, tile.rms, tile.zcr);
}

// Or from a WAV file:
let result = decomposer.decompose_wav(&wav_bytes)?;

// Reassemble:
let original = AudioDecomposer::reassemble(&result.tiles);
```

## Spectral Features

| Feature | Description |
|---------|-------------|
| `spectral_peak` | Estimated dominant frequency (Hz) via zero-crossing intervals |
| `rms` | Root mean square energy — loudness indicator |
| `zcr` | Zero-crossing rate — noisiness/brightness indicator |

## How It Feeds Plato Agents

1. **Ingest**: Raw audio (WAV files, streamed samples) enters the pipeline
2. **Decompose**: `forge-soniqo` splits into chunks with spectral metadata
3. **Store**: Audio tiles go into `forge-memory` for persistence
4. **Query**: Agents can search by spectral characteristics (e.g., "find loud segments")

## License

MIT
