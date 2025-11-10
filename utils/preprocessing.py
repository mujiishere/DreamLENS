import numpy as np
import scipy.signal as signal
import scipy.io as sio
import mne
from scipy import stats
import torch

class EEGPreprocessor:
    """
    EEG Data Preprocessing Module
    """
    
    def __init__(self, sampling_rate=256, lowcut=0.5, highcut=45.0, notch_freq=50.0):
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        
    def load_eeg_data(self, file_path, file_type='mat'):
        """Load EEG data from .mat or .csv files"""
        if file_type == 'mat':
            data = sio.loadmat(file_path)
            eeg_data = data['eeg_data']  # Shape: (channels, time_points)
        elif file_type == 'csv':
            eeg_data = np.loadtxt(file_path, delimiter=',')
        else:
            raise ValueError("Unsupported file type. Use 'mat' or 'csv'")
            
        return eeg_data
    
    def apply_bandpass_filter(self, eeg_data):
        """Apply bandpass filter to EEG data"""
        nyquist = self.sampling_rate / 2.0
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        filtered_data = np.zeros_like(eeg_data)
        for i in range(eeg_data.shape[0]):
            filtered_data[i] = signal.filtfilt(b, a, eeg_data[i])
            
        return filtered_data
    
    def apply_notch_filter(self, eeg_data):
        """Apply notch filter to remove power line noise"""
        nyquist = self.sampling_rate / 2.0
        freq = self.notch_freq / nyquist
        
        b, a = signal.iirnotch(freq, 30)
        
        filtered_data = np.zeros_like(eeg_data)
        for i in range(eeg_data.shape[0]):
            filtered_data[i] = signal.filtfilt(b, a, eeg_data[i])
            
        return filtered_data
    
    def normalize_eeg(self, eeg_data):
        """Normalize EEG data using z-score normalization"""
        normalized_data = np.zeros_like(eeg_data)
        for i in range(eeg_data.shape[0]):
            normalized_data[i] = stats.zscore(eeg_data[i])
            
        return normalized_data
    
    def preprocess_pipeline(self, file_path, file_type='mat'):
        """Complete preprocessing pipeline"""
        eeg_data = self.load_eeg_data(file_path, file_type)
        eeg_data = self.apply_notch_filter(eeg_data)
        eeg_data = self.apply_bandpass_filter(eeg_data)
        eeg_data = self.normalize_eeg(eeg_data)
        
        return eeg_data