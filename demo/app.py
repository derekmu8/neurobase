"""
Streamlit Demo App for NeuroBase Seizure Detection

Run with: streamlit run demo/app.py
"""
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from models.encoder import EEGEncoder, create_encoder_from_config
from models.classifier import SeizureClassifier
from data.preprocessing import preprocess_edf, load_edf, normalize_channels


# Page config
st.set_page_config(
    page_title="NeuroBase - EEG Seizure Detection",
    page_icon="🧠",
    layout="wide"
)


@st.cache_resource
def load_model():
    """Load the trained model"""
    checkpoint_path = Path("checkpoints/best_classifier.pt")
    
    # Create model
    encoder = create_encoder_from_config(config)
    classifier = SeizureClassifier(encoder, freeze_encoder=False)
    
    # Load weights if available
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        st.sidebar.success("✅ Model loaded from checkpoint")
    else:
        st.sidebar.warning("⚠️ No checkpoint found, using random weights")
    
    classifier.eval()
    return classifier


@st.cache_data
def load_sample_data():
    """Load pre-loaded sample EEG data"""
    samples_dir = Path("demo/samples")
    samples = {}
    
    # Check for sample files
    if samples_dir.exists():
        for sample_file in samples_dir.glob("*.npy"):
            name = sample_file.stem
            data = np.load(sample_file, allow_pickle=True).item()
            samples[name] = data
    
    # If no samples exist, create synthetic ones
    if not samples:
        np.random.seed(42)
        
        # Normal EEG-like signal
        t = np.linspace(0, 4, 1024)
        normal_signal = np.zeros((18, 1024))
        for ch in range(18):
            # Mix of alpha (10Hz) and beta (20Hz) rhythms
            normal_signal[ch] = (
                0.5 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi) +
                0.3 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi) +
                0.2 * np.random.randn(1024)
            )
        samples['normal_eeg'] = {
            'signal': normal_signal.astype(np.float32),
            'label': 0,
            'description': 'Normal EEG activity'
        }
        
        # Seizure-like signal (high amplitude, rhythmic spikes)
        seizure_signal = np.zeros((18, 1024))
        for ch in range(18):
            # Seizure pattern: high amplitude rhythmic activity
            base = 2.0 * np.sin(2 * np.pi * 3 * t)  # 3Hz spike-wave
            spikes = np.zeros(1024)
            spike_times = np.arange(0, 1024, 85)  # Regular spikes
            for st in spike_times:
                if st + 20 < 1024:
                    spikes[st:st+20] = 3.0 * np.exp(-np.linspace(0, 3, 20))
            seizure_signal[ch] = base + spikes + 0.3 * np.random.randn(1024)
        
        samples['seizure_eeg'] = {
            'signal': seizure_signal.astype(np.float32),
            'label': 1,
            'description': 'Seizure activity (3Hz spike-wave pattern)'
        }
    
    return samples


def plot_eeg_signal(signal: np.ndarray, attention: np.ndarray = None, title: str = "EEG Signal"):
    """
    Plot EEG signal with optional attention overlay.
    
    Args:
        signal: (n_channels, n_samples)
        attention: (n_patches,) attention weights
        title: Plot title
    """
    n_channels, n_samples = signal.shape
    
    fig, axes = plt.subplots(figsize=(14, 8))
    
    # Calculate offset for stacked channels
    channel_spacing = 3.0
    
    # Time axis
    time = np.linspace(0, 4, n_samples)  # 4 seconds
    
    # Channel names
    channel_names = config.data.common_channels[:n_channels]
    
    # Plot each channel
    for ch in range(n_channels):
        offset = (n_channels - ch - 1) * channel_spacing
        axes.plot(time, signal[ch] + offset, 'b-', linewidth=0.5, alpha=0.8)
    
    # Add attention heatmap if provided
    if attention is not None:
        # Normalize attention
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        # Create patches for attention
        n_patches = len(attention)
        patch_duration = 4.0 / n_patches
        
        for i, attn_val in enumerate(attention):
            start_time = i * patch_duration
            end_time = (i + 1) * patch_duration
            
            # Color intensity based on attention
            color = plt.cm.Reds(attn_val)
            axes.axvspan(start_time, end_time, alpha=attn_val * 0.3, color='red')
    
    # Labels
    axes.set_xlabel('Time (s)', fontsize=12)
    axes.set_ylabel('Channels', fontsize=12)
    axes.set_title(title, fontsize=14)
    
    # Y-axis ticks for channels
    axes.set_yticks([(n_channels - ch - 1) * channel_spacing for ch in range(n_channels)])
    axes.set_yticklabels(channel_names, fontsize=8)
    
    axes.set_xlim(0, 4)
    axes.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_attention_bar(attention: np.ndarray):
    """Plot attention weights as a bar chart"""
    fig, ax = plt.subplots(figsize=(12, 2))
    
    n_patches = len(attention)
    x = np.arange(n_patches)
    
    # Normalize
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    
    colors = plt.cm.Reds(attention)
    ax.bar(x, attention, color=colors, edgecolor='darkred', linewidth=0.5)
    
    # Labels
    ax.set_xlabel('Time Patch (0.25s each)', fontsize=10)
    ax.set_ylabel('Attention', fontsize=10)
    ax.set_title('Model Attention over Time', fontsize=12)
    ax.set_xlim(-0.5, n_patches - 0.5)
    ax.set_ylim(0, 1.1)
    
    # Time labels
    patch_times = [f"{i*0.25:.2f}s" for i in range(0, n_patches, 4)]
    ax.set_xticks(range(0, n_patches, 4))
    ax.set_xticklabels(patch_times)
    
    plt.tight_layout()
    return fig


def main():
    st.title("🧠 NeuroBase - EEG Seizure Detection")
    st.markdown("""
    **Foundation model for EEG analysis** - Transformer-based seizure detection with attention visualization.
    """)
    
    # Sidebar
    st.sidebar.header("Model Info")
    
    # Load model
    model = load_model()
    
    st.sidebar.markdown(f"""
    - **Architecture**: Transformer Encoder
    - **Parameters**: {sum(p.numel() for p in model.parameters()):,}
    - **Input**: 18 channels × 4 seconds
    - **Patch Size**: 64 samples (0.25s)
    """)
    
    # Load samples
    samples = load_sample_data()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Input Signal")
        
        # Sample selection
        sample_name = st.selectbox(
            "Select sample EEG:",
            list(samples.keys()),
            format_func=lambda x: f"{x} - {samples[x]['description']}"
        )
        
        sample = samples[sample_name]
        signal = sample['signal']
        true_label = sample['label']
        
        st.info(f"**Ground Truth**: {'🔴 Seizure' if true_label == 1 else '🟢 Non-Seizure'}")
        
        # Plot signal
        st.pyplot(plot_eeg_signal(signal, title=f"EEG Signal: {sample_name}"))
    
    with col2:
        st.header("Prediction")
        
        # Run inference
        with torch.no_grad():
            x = torch.from_numpy(signal).unsqueeze(0).float()
            output = model(x, return_attention=True)
            prob = output['probs'].item()
            
            # Get attention weights
            attention = model.get_attention_weights(x)
            if attention is not None:
                attention = attention.squeeze().numpy()
        
        # Display prediction
        st.markdown("### Seizure Probability")
        
        # Large probability display
        if prob > 0.5:
            st.error(f"## 🔴 {prob*100:.1f}%")
            prediction = "SEIZURE DETECTED"
        else:
            st.success(f"## 🟢 {(1-prob)*100:.1f}%")
            prediction = "NO SEIZURE"
        
        st.markdown(f"**Prediction**: {prediction}")
        
        # Confidence bar
        st.progress(prob)
        
        # Metrics
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Confidence", f"{max(prob, 1-prob)*100:.1f}%")
        with col_b:
            st.metric("Threshold", "50%")
    
    # Attention visualization
    st.header("Model Attention")
    st.markdown("""
    The attention heatmap shows which parts of the signal the model focused on most when making its prediction.
    Higher attention (red) indicates regions the model considered most important.
    """)
    
    if attention is not None:
        # Attention bar chart
        st.pyplot(plot_attention_bar(attention))
        
        # Signal with attention overlay
        st.pyplot(plot_eeg_signal(signal, attention=attention, title="Signal with Attention Overlay"))
    else:
        st.warning("Attention visualization not available")
    
    # Technical details (expandable)
    with st.expander("Technical Details"):
        st.markdown(f"""
        ### Model Architecture
        - **Encoder**: Transformer with {config.model.n_layers} layers
        - **Hidden Dimension**: {config.model.d_model}
        - **Attention Heads**: {config.model.n_heads}
        - **Patches**: {config.model.n_patches} patches of {config.model.patch_size} samples
        
        ### Input Processing
        - **Channels**: {config.model.n_channels}
        - **Sampling Rate**: {config.data.sfreq} Hz
        - **Window Duration**: {config.data.window_sec} seconds
        - **Normalization**: Z-score per channel
        
        ### Training
        - **Pretraining**: Masked patch prediction (40% mask ratio)
        - **Fine-tuning**: Binary cross-entropy loss
        - **Dataset**: CHB-MIT Scalp EEG Database
        """)
        
        if attention is not None:
            st.markdown("### Attention Statistics")
            st.write(f"- Max attention: {attention.max():.4f}")
            st.write(f"- Min attention: {attention.min():.4f}")
            st.write(f"- Mean attention: {attention.mean():.4f}")
            st.write(f"- Peak attention patch: {attention.argmax()} ({attention.argmax() * 0.25:.2f}s)")


if __name__ == "__main__":
    main()
