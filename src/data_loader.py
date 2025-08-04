import torch
import numpy as np
import scipy.io as sio
import os
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from sklearn.preprocessing import StandardScaler
import random

class EEGPreprocessor:
    """EEG preprocessing pipeline as described in the paper"""
    
    def __init__(self, sampling_rate=200, band_pass_low=1, band_pass_high=50):
        self.sampling_rate = sampling_rate
        self.band_pass_low = band_pass_low
        self.band_pass_high = band_pass_high
        
    def band_pass_filter(self, eeg_data):
        """Apply band-pass filter (1-50 Hz) using zero-phase Butterworth filter"""
        nyquist = self.sampling_rate / 2
        low = self.band_pass_low / nyquist
        high = self.band_pass_high / nyquist
        
        # Design Butterworth filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply zero-phase filtering
        filtered_data = signal.filtfilt(b, a, eeg_data, axis=-1)
        
        return filtered_data
    
    def z_score_normalization(self, eeg_data):
        """Channel-wise z-score normalization"""
        # Calculate mean and std for each channel
        mean_vals = np.mean(eeg_data, axis=-1, keepdims=True)
        std_vals = np.std(eeg_data, axis=-1, keepdims=True)
        
        # Avoid division by zero
        std_vals = np.where(std_vals == 0, 1, std_vals)
        
        # Apply z-score normalization
        normalized_data = (eeg_data - mean_vals) / std_vals
        
        return normalized_data
    
    def segment_data(self, eeg_data, segment_length=800):
        """Segment EEG data into 4-second epochs"""
        # segment_length = 4 seconds * 200 Hz = 800 samples
        segments = []
        
        for i in range(0, eeg_data.shape[-1] - segment_length + 1, segment_length):
            segment = eeg_data[:, i:i+segment_length]
            segments.append(segment)
        
        return np.array(segments)
    
    def apply_augmentation(self, eeg_data, temporal_jitter_prob=0.3, noise_prob=0.4):
        """Apply data augmentation techniques"""
        augmented_data = eeg_data.copy()
        
        # Temporal jittering (±250ms = ±50 samples at 200Hz)
        if random.random() < temporal_jitter_prob:
            jitter = random.randint(-50, 50)
            if jitter > 0:
                augmented_data = np.pad(augmented_data, ((0, 0), (0, 0), (jitter, 0)), mode='edge')
                augmented_data = augmented_data[:, :, :-jitter]
            else:
                augmented_data = np.pad(augmented_data, ((0, 0), (0, 0), (0, -jitter)), mode='edge')
                augmented_data = augmented_data[:, :, -jitter:]
        
        # Gaussian noise addition
        if random.random() < noise_prob:
            noise_std = 0.01 * np.std(augmented_data)
            noise = np.random.normal(0, noise_std, augmented_data.shape)
            augmented_data = augmented_data + noise
        
        return augmented_data

class EEGDataset(Dataset):
    """EEG Emotion Recognition Dataset"""
    
    def __init__(self, data_path, dataset_name='SEED', transform=None, preprocessor=None):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.transform = transform
        self.preprocessor = preprocessor or EEGPreprocessor()
        
        self.data, self.labels = self._load_data()
        
    def _load_data(self):
        """Load EEG data and labels"""
        if self.dataset_name == 'SEED':
            return self._load_seed_data()
        elif self.dataset_name == 'SEED-V':
            return self._load_seedv_data()
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
    
    def _load_seed_data(self):
        """Load SEED dataset (15 subjects, 3 emotions)"""
        data = []
        labels = []
        
        # SEED: 15 subjects, 3 emotions (positive, negative, neutral)
        for subject in range(1, 16):
            for session in range(1, 4):  # 3 sessions
                for emotion in range(3):  # 3 emotions
                    file_path = os.path.join(self.data_path, f'SEED', f'{subject}_{session}_{emotion}.mat')
                    if os.path.exists(file_path):
                        try:
                            mat_data = sio.loadmat(file_path)
                            # Assuming the EEG data is stored in 'eeg_data' field
                            eeg_data = mat_data.get('eeg_data', mat_data.get('data', None))
                            
                            if eeg_data is not None:
                                # Preprocess the data
                                eeg_data = self.preprocessor.band_pass_filter(eeg_data)
                                eeg_data = self.preprocessor.z_score_normalization(eeg_data)
                                
                                # Segment into 4-second epochs
                                segments = self.preprocessor.segment_data(eeg_data)
                                
                                data.extend(segments)
                                labels.extend([emotion] * len(segments))
                                
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
                            continue
        
        return np.array(data), np.array(labels)
    
    def _load_seedv_data(self):
        """Load SEED-V dataset (20 subjects, 5 emotions)"""
        data = []
        labels = []
        
        # SEED-V: 20 subjects, 5 emotions (happiness, sadness, fear, disgust, neutral)
        for subject in range(1, 21):
            for session in range(1, 4):  # 3 sessions
                for emotion in range(5):  # 5 emotions
                    file_path = os.path.join(self.data_path, f'SEED-V', f'{subject}_{session}_{emotion}.mat')
                    if os.path.exists(file_path):
                        try:
                            mat_data = sio.loadmat(file_path)
                            eeg_data = mat_data.get('eeg_data', mat_data.get('data', None))
                            
                            if eeg_data is not None:
                                # Preprocess the data
                                eeg_data = self.preprocessor.band_pass_filter(eeg_data)
                                eeg_data = self.preprocessor.z_score_normalization(eeg_data)
                                
                                # Segment into 4-second epochs
                                segments = self.preprocessor.segment_data(eeg_data)
                                
                                data.extend(segments)
                                labels.extend([emotion] * len(segments))
                                
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
                            continue
        
        return np.array(data), np.array(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        eeg_data = self.data[idx]
        label = self.labels[idx]
        
        # Apply data augmentation during training
        if self.transform:
            eeg_data = self.preprocessor.apply_augmentation(eeg_data)
        
        return torch.FloatTensor(eeg_data), torch.LongTensor([label])

def create_loso_data_loaders(data_path, dataset_name='SEED', batch_size=32, subject_id=None):
    """Create Leave-One-Subject-Out (LOSO) data loaders"""
    
    # Load all data
    dataset = EEGDataset(data_path, dataset_name, transform=True)
    
    if subject_id is None:
        # Return all data for training
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = None
    else:
        # Split data for LOSO validation
        # This is a simplified version - in practice, you'd need to track subject IDs
        total_samples = len(dataset)
        val_size = total_samples // 15 if dataset_name == 'SEED' else total_samples // 20
        
        # Simple split (in practice, you'd split by actual subject)
        train_size = total_samples - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def create_frequency_band_data(eeg_data, sampling_rate=200):
    """Extract different frequency bands for analysis"""
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    band_data = {}
    
    for band_name, (low_freq, high_freq) in bands.items():
        # Design band-pass filter
        nyquist = sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, eeg_data, axis=-1)
        
        band_data[band_name] = filtered_data
    
    return band_data

if __name__ == "__main__":
    # Test the data loader
    preprocessor = EEGPreprocessor()
    
    # Create dummy EEG data for testing
    dummy_eeg = np.random.randn(62, 2400)  # 62 channels, 12 seconds at 200Hz
    
    # Test preprocessing
    filtered = preprocessor.band_pass_filter(dummy_eeg)
    normalized = preprocessor.z_score_normalization(filtered)
    segments = preprocessor.segment_data(normalized)
    
    print(f"Original shape: {dummy_eeg.shape}")
    print(f"Filtered shape: {filtered.shape}")
    print(f"Normalized shape: {normalized.shape}")
    print(f"Number of segments: {len(segments)}")
    print(f"Segment shape: {segments[0].shape}") 