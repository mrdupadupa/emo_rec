import os
import numpy as np
import tensorflow.compat.v1 as tf
from magenta.models.music_vae import configs
from magenta.models.music_vae import TrainedModel
import note_seq

# Disable TensorFlow 2.x behavior
tf.disable_v2_behavior()

# Define emotion-specific parameters based on available models
EMOTION_PARAMS = {
    'happy': {
        'config': 'cat-mel_2bar_big',  # Melodic content for happy emotions
        'temperature': 0.7,            # Higher temperature for creative, vibrant patterns
        'num_samples': 10,
    },
    'sad': {
        'config': 'cat-mel_2bar_big',  # Melodic content but with lower temperature
        'temperature': 0.3,            # Lower temperature for more predictable, melancholic patterns
        'num_samples': 10,
    },
    'angry': {
        'config': 'nade-drums_2bar_full',  # Full percussion for intense emotions
        'temperature': 0.8,                # High temperature for chaotic patterns
        'num_samples': 10,
    },
    'fear': {
        'config': 'cat-drums_2bar_small',  # Simpler percussion for tense emotions
        'temperature': 0.4,                # Moderate-low temperature for controlled unpredictability
        'num_samples': 10,
    },
    'surprise': {
        'config': 'hierdec-mel_16bar',     # Longer melodic phrases for surprise
        'temperature': 0.9,                # High temperature for unexpected variations
        'num_samples': 10,
    },
    'disgust': {
        'config': 'nade-drums_2bar_full',  # Full percussion for negative emotions
        'temperature': 0.6,                # Moderate-high temperature
        'num_samples': 10,
    },
    'neutral': {
        'config': 'cat-mel_2bar_big',      # Melodic content for neutral state
        'temperature': 0.5,                # Balanced temperature
        'num_samples': 10,
    },
    # Blended emotions for transitions
    'happy_sad': {
        'config': 'cat-mel_2bar_big',
        'temperature': 0.5,                # Halfway between happy and sad temperatures
        'num_samples': 5,
    },
    'happy_angry': {
        'config': 'cat-mel_2bar_big',      # Using melodic content instead of percussion
        'temperature': 0.75,               # Balanced toward happy
        'num_samples': 5,
    },
    'sad_fear': {
        'config': 'cat-mel_2bar_big',
        'temperature': 0.35,               # Low temperature for negative emotions
        'num_samples': 5,
    },
    'surprise_happy': {
        'config': 'hierdec-mel_16bar',     # Longer sequences for complex emotion blend
        'temperature': 0.8,
        'num_samples': 5,
    }
}

# Path to your checkpoints
CHECKPOINT_DIR = "/home/andrei/Work/emotion_recognition/llm+emo_rec/magenta/"

# Output directory for generated samples
OUTPUT_DIR = "emotion_samples"

def generate_emotion_samples():
    """Generate and save emotion-based music samples."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # For each emotion, generate samples
    for emotion, params in EMOTION_PARAMS.items():
        print(f"Generating samples for {emotion}...")
        
        # Create emotion-specific directory
        emotion_dir = os.path.join(OUTPUT_DIR, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        # Load the appropriate model configuration
        config_name = params['config']
        config = configs.CONFIG_MAP[config_name]
        
        # Create the model
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{config_name}.tar")
        model = TrainedModel(
            config=config,
            batch_size=4,  # Small batch for memory efficiency
            checkpoint_dir_or_path=checkpoint_path
        )
        
        # Generate samples
        for i in range(params['num_samples']):
            print(f"  Sample {i+1}/{params['num_samples']}")
            
            # Generate a sample
            samples = model.sample(
                n=1,  # Generate one sample at a time
                length=config.hparams.max_seq_len,
                temperature=params['temperature']
            )
            
            # Save the sample as a MIDI file
            output_path = os.path.join(emotion_dir, f"{emotion}_{i+1}.mid")
            note_seq.sequence_proto_to_midi_file(samples[0], output_path)
            
            print(f"  Saved to {output_path}")
    
    print("Done generating all emotion samples!")

if __name__ == "__main__":
    generate_emotion_samples()