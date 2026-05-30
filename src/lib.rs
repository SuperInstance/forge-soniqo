//! # forge-soniqo
//!
//! Audio decomposition for Plato agents. Splits audio into chunks with spectral
//! metadata. Computes peak frequency, RMS, and zero-crossing rate from raw samples
//! with zero external audio dependencies.

use serde::{Deserialize, Serialize};

/// A decomposer that splits raw audio samples into chunks.
#[derive(Debug, Clone)]
pub struct AudioDecomposer {
    pub sample_rate: u32,
    pub chunk_duration_ms: u32,
}

/// A single audio tile with spectral features.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioTile {
    pub samples: Vec<f32>,
    pub spectral_peak: f64,
    pub rms: f64,
    pub zcr: f64,
    pub timestamp_ms: u64,
}

/// Result of decomposing audio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionResult {
    pub tiles: Vec<AudioTile>,
    pub sample_rate: u32,
    pub total_duration_ms: u64,
    pub chunk_duration_ms: u32,
}

impl AudioDecomposer {
    /// Create a new decomposer with the given sample rate and chunk duration.
    pub fn new(sample_rate: u32, chunk_duration_ms: u32) -> Self {
        Self { sample_rate, chunk_duration_ms }
    }

    /// Samples per chunk based on sample rate and chunk duration.
    pub fn samples_per_chunk(&self) -> usize {
        (self.sample_rate as u64 * self.chunk_duration_ms as u64 / 1000) as usize
    }

    /// Decompose raw samples into audio tiles with spectral features.
    pub fn decompose(&self, samples: &[f32]) -> DecompositionResult {
        let chunk_size = self.samples_per_chunk();
        let mut tiles = Vec::new();

        let mut offset = 0;
        while offset < samples.len() {
            let end = std::cmp::min(offset + chunk_size, samples.len());
            let chunk = &samples[offset..end];

            let rms = compute_rms(chunk);
            let zcr = compute_zcr(chunk);
            let spectral_peak = compute_spectral_peak(chunk, self.sample_rate);

            let timestamp_ms = (offset as u64 * 1000) / self.sample_rate as u64;

            tiles.push(AudioTile {
                samples: chunk.to_vec(),
                spectral_peak,
                rms,
                zcr,
                timestamp_ms,
            });

            offset = end;
        }

        let total_duration_ms = if samples.is_empty() {
            0
        } else {
            (samples.len() as u64 * 1000) / self.sample_rate as u64
        };

        DecompositionResult {
            tiles,
            sample_rate: self.sample_rate,
            total_duration_ms,
            chunk_duration_ms: self.chunk_duration_ms,
        }
    }

    /// Reassemble tiles back into a single sample buffer.
    pub fn reassemble(tiles: &[AudioTile]) -> Vec<f32> {
        let mut result = Vec::new();
        for tile in tiles {
            result.extend_from_slice(&tile.samples);
        }
        result
    }

    /// Decompose a WAV file from raw bytes. Handles basic 16-bit PCM WAV.
    pub fn decompose_wav(&self, wav_data: &[u8]) -> Result<DecompositionResult, String> {
        let samples = parse_wav(wav_data)?;
        Ok(self.decompose(&samples))
    }
}

/// Compute RMS (root mean square) of samples.
pub fn compute_rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

/// Compute zero-crossing rate.
pub fn compute_zcr(samples: &[f32]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let crossings = samples.windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    crossings as f64 / (samples.len() - 1) as f64
}

/// Compute spectral peak frequency using zero-crossing estimation.
/// A simple approach: estimate dominant frequency from zero-crossing intervals.
pub fn compute_spectral_peak(samples: &[f32], sample_rate: u32) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }

    // Collect intervals between zero crossings (in samples)
    let mut intervals = Vec::new();
    let mut last_cross = 0usize;

    for i in 1..samples.len() {
        if (samples[i - 1] >= 0.0) != (samples[i] >= 0.0) {
            if last_cross > 0 {
                intervals.push(i - last_cross);
            }
            last_cross = i;
        }
    }

    if intervals.is_empty() {
        return 0.0;
    }

    // Median interval gives robust frequency estimate
    intervals.sort();
    let median_interval = intervals[intervals.len() / 2] as f64;

    // Each full cycle = 2 zero crossings, so frequency = sr / (2 * half_interval)
    // But intervals here are between consecutive crossings, so cycle = 2 * interval
    sample_rate as f64 / (2.0 * median_interval)
}

/// Parse a simple 16-bit PCM WAV file into f32 samples normalized to [-1, 1].
pub fn parse_wav(data: &[u8]) -> Result<Vec<f32>, String> {
    if data.len() < 44 {
        return Err("WAV data too short".into());
    }

    // Check RIFF header
    if &data[0..4] != b"RIFF" {
        return Err("Not a RIFF file".into());
    }
    if &data[8..12] != b"WAVE" {
        return Err("Not a WAVE file".into());
    }

    // Parse fmt chunk
    let num_channels = u16::from_le_bytes([data[22], data[23]]);
    let _sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    let bits_per_sample = u16::from_le_bytes([data[34], data[35]]);

    if bits_per_sample != 16 {
        return Err(format!("Unsupported bits per sample: {}", bits_per_sample));
    }

    // Find data chunk
    let mut offset = 12usize;
    while offset + 8 <= data.len() {
        let chunk_id = &data[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as usize;

        if chunk_id == b"data" {
            let data_start = offset + 8;
            let data_end = std::cmp::min(data_start + chunk_size, data.len());
            let _num_samples = (data_end - data_start) / 2;

            let samples: Vec<f32> = data[data_start..data_end]
                .chunks_exact(2)
                .map(|c| {
                    let raw = i16::from_le_bytes([c[0], c[1]]);
                    raw as f32 / 32768.0
                })
                .collect();

            // If stereo, take first channel
            if num_channels == 2 {
                return Ok(samples.into_iter().step_by(2).collect());
            }

            return Ok(samples);
        }
        offset += 8 + chunk_size;
        // Align to even
        if !offset.is_multiple_of(2) {
            offset += 1;
        }
    }

    Err("No data chunk found".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple 16-bit mono WAV with a sine wave.
    fn make_sine_wav(freq: f64, duration_secs: f64, sample_rate: u32) -> Vec<u8> {
        let num_samples = (sample_rate as f64 * duration_secs) as usize;
        let samples: Vec<i16> = (0..num_samples)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                let val = (2.0 * std::f64::consts::PI * freq * t).sin();
                (val * 32000.0) as i16
            })
            .collect();

        let data_size = num_samples * 2;
        let file_size = 36 + data_size;
        let mut wav = Vec::with_capacity(44 + data_size);

        // RIFF header
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(file_size as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");

        // fmt chunk
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM format
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data_size as u32).to_le_bytes());
        for s in samples {
            wav.extend_from_slice(&s.to_le_bytes());
        }

        wav
    }

    #[test]
    fn test_decompose_basic() {
        let decomposer = AudioDecomposer::new(44100, 100);
        let samples: Vec<f32> = (0..44100).map(|i| (i as f32 / 44100.0).sin() * 0.5).collect();
        let result = decomposer.decompose(&samples);

        assert_eq!(result.tiles.len(), 10); // 1 second / 100ms chunks
        assert_eq!(result.total_duration_ms, 1000);
        assert_eq!(result.sample_rate, 44100);
    }

    #[test]
    fn test_chunk_sizing() {
        let decomposer = AudioDecomposer::new(16000, 50);
        assert_eq!(decomposer.samples_per_chunk(), 800);

        let samples: Vec<f32> = vec![0.0; 1600];
        let result = decomposer.decompose(&samples);
        assert_eq!(result.tiles.len(), 2);
    }

    #[test]
    fn test_spectral_features() {
        // Generate a sine wave at known frequency
        let sample_rate = 44100u32;
        let freq = 440.0f64;
        let samples: Vec<f32> = (0..4410)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (2.0 * std::f64::consts::PI * freq * t).sin() as f32
            })
            .collect();

        let decomposer = AudioDecomposer::new(sample_rate, 100);
        let result = decomposer.decompose(&samples);

        let tile = &result.tiles[0];
        assert!(tile.rms > 0.0);
        assert!(tile.zcr > 0.0);
        // Spectral peak should be roughly near 440 Hz (within tolerance)
        assert!(tile.spectral_peak > 200.0 && tile.spectral_peak < 800.0,
            "spectral_peak was {}", tile.spectral_peak);
    }

    #[test]
    fn test_rms_silence() {
        let rms = compute_rms(&[0.0, 0.0, 0.0]);
        assert_eq!(rms, 0.0);
    }

    #[test]
    fn test_rms_full_scale() {
        let rms = compute_rms(&[1.0, -1.0, 1.0, -1.0]);
        assert!((rms - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_zcr_dc() {
        // Constant signal = no crossings
        let zcr = compute_zcr(&[0.5, 0.5, 0.5, 0.5]);
        assert_eq!(zcr, 0.0);
    }

    #[test]
    fn test_zcr_alternating() {
        let zcr = compute_zcr(&[1.0, -1.0, 1.0, -1.0]);
        assert!(zcr > 0.5);
    }

    #[test]
    fn test_reassembly() {
        let decomposer = AudioDecomposer::new(44100, 50);
        let original: Vec<f32> = (0..8820).map(|i| (i as f32 * 0.001).sin()).collect();
        let result = decomposer.decompose(&original);
        let reassembled = AudioDecomposer::reassemble(&result.tiles);

        assert_eq!(original.len(), reassembled.len());
        for (a, b) in original.iter().zip(reassembled.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_decompose_wav() {
        let wav = make_sine_wav(440.0, 0.5, 44100);
        let decomposer = AudioDecomposer::new(44100, 100);
        let result = decomposer.decompose_wav(&wav).unwrap();
        assert!(result.tiles.len() >= 5);
        assert!(result.total_duration_ms >= 400);
    }

    #[test]
    fn test_empty_samples() {
        let decomposer = AudioDecomposer::new(44100, 100);
        let result = decomposer.decompose(&[]);
        assert!(result.tiles.is_empty());
        assert_eq!(result.total_duration_ms, 0);
    }

    #[test]
    fn test_timestamps() {
        let decomposer = AudioDecomposer::new(44100, 100);
        let samples: Vec<f32> = vec![0.0; 44100];
        let result = decomposer.decompose(&samples);

        assert_eq!(result.tiles[0].timestamp_ms, 0);
        assert_eq!(result.tiles[1].timestamp_ms, 100);
        assert_eq!(result.tiles[5].timestamp_ms, 500);
    }
}
