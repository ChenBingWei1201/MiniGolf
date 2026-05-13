"""
Brain-Computer Interface MLP Classifier (BrainLink Version)
For EEG signal relaxation/focus/blink state classification
USE RAW DATA AS INPUT
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import os
import glob
import warnings
warnings.filterwarnings('ignore')
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import skew, kurtosis
import joblib

# Parameter Settings
class Config:
    # Dataset path settings
    DATASET_PATH = "bci_dataset_114-2_any"
    
    # MLP model parameters
    HIDDEN_LAYERS = (128, 64, 32) #e.g. (128, 64, 32), (256, 128)
    MAX_ITER = 100 #50 ~ 200
    LEARNING_RATE = 0.005 #0.005 ~ 0.02
    ALPHA = 0.001 #0.0001 ~ 0.05
    ACTIVATION = 'relu'
    SOLVER = 'adam'
    BATCH_SIZE = 64 #32 ~ 128
    EARLY_STOPPING = True
    VALIDATION_FRACTION = 0.1
    N_ITER_NO_CHANGE = 10
    
    # Signal processing parameters
    SAMPLING_RATE = 512    # BrainLink fixed sampling rate
    SEGMENT_LENGTH = 5     # Segment length in seconds  2 ~ 6
    OVERLAP_RATIO = 0.6    # Overlap ratio for segments  0.0 ~ 0.8
    
    # Feature selection parameters
    FEATURE_SELECTION = True
    N_FEATURES_SELECT = 8 # Modify preprocessing to extract truly effective features
    
    # Other settings
    RANDOM_STATE = 42

def create_segments(data, segment_length_samples, overlap_samples):
    """Split a single round of continuous EEG signal into multiple segments"""
    if len(data) < segment_length_samples:
        return []
    
    segments = []
    start = 0
    step = segment_length_samples - overlap_samples
    
    while start + segment_length_samples <= len(data):
        segment = data[start:start + segment_length_samples]
        
        # ==========================================
        # === STUDENT PREPROCESSING HERE (Part 1)===
        # ==========================================
        
        # === student preprocessing ===
        #BPF
        order=4
        fL=1.0
        fH=40.0
        sos=butter(order,[fL,fH],btype='bandpass',fs=Config.SAMPLING_RATE,output='sos')
        segment=sosfiltfilt(sos,segment)
        # === student preprocessing ===

        segments.append(segment)
        start += step
    
    return segments

def extract_features(segments):
    """
    Perform feature engineering on segments
    """
    features = []
    for seg in segments:
        # ==========================================
        # === STUDENT PREPROCESSING HERE (Part 2)===
        # ==========================================
        
        # === student preprocessing ===
        var=np.var(seg)
        ptp=np.ptp(seg)
        rms=np.sqrt(np.mean(seg**2))
        freqs,psd=welch(seg,fs=Config.SAMPLING_RATE,nperseg=int(Config.SAMPLING_RATE/2))
        delta=np.sum(psd[(freqs>=1)&(freqs<4)])
        theta=np.sum(psd[(freqs>=4)&(freqs<8)])
        alpha=np.sum(psd[(freqs>=8)&(freqs<13)])
        beta=np.sum(psd[(freqs>=13)&(freqs<=30)])
        total_power=delta+theta+alpha+beta

        current_feature = np.array([
            var,
            ptp,
            rms,
            delta/total_power,theta/total_power,alpha/total_power,beta/total_power,alpha/beta
        ])
        
        #current_feature = seg 
        
        # === student preprocessing ===
        
        features.append(current_feature)
        
    return np.array(features)

def load_all_subjects():
    """Load round-based data for all subjects in the group"""
    all_features = []
    all_labels = []
    all_subjects = []
    
    if not os.path.exists(Config.DATASET_PATH):
        print(f"Error: Directory '{Config.DATASET_PATH}' not found")
        return None, None, None
        
    subject_folders = sorted([f.path for f in os.scandir(Config.DATASET_PATH) if f.is_dir()])
    
    if len(subject_folders) < 2:
        print("Error: Not enough subjects. At least 2 subject folders are required for cross-validation.")
        return None, None, None
        
    print(f"Found {len(subject_folders)} subjects. Loading data...")
    
    segment_length_samples = int(Config.SEGMENT_LENGTH * Config.SAMPLING_RATE)
    overlap_samples = int(segment_length_samples * Config.OVERLAP_RATIO)

    for subject_folder in subject_folders:
        subject_id = os.path.basename(subject_folder)
        relax_segments = []
        focus_segments = []
        blink_segments = []
        
        # Load Task 1 (Relax) all rounds
        task1_files = glob.glob(os.path.join(subject_folder, "*_1_*.txt"))
        for file in task1_files:
            try:
                data = np.loadtxt(file)
                segs = create_segments(data, segment_length_samples, overlap_samples)
                relax_segments.extend(segs)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        # Load Task 2 (Focus) all rounds
        task2_files = glob.glob(os.path.join(subject_folder, "*_2_*.txt"))
        for file in task2_files:
            try:
                data = np.loadtxt(file)
                segs = create_segments(data, segment_length_samples, overlap_samples)
                focus_segments.extend(segs)
            except Exception as e:
                print(f"Error reading {file}: {e}")
        
        # Load Task 3 (Blink) all rounds
        task3_files = glob.glob(os.path.join(subject_folder, "*_3_*.txt"))
        for file in task3_files:
            try:
                data = np.loadtxt(file)
                segs = create_segments(data, segment_length_samples, overlap_samples)
                blink_segments.extend(segs)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        if len(relax_segments) == 0 or len(focus_segments) == 0 or len(blink_segments) == 0:
            print(f"Warning: Insufficient data for {subject_id}. Skipping.")
            continue

        # Extract features
        relax_features = extract_features(relax_segments)
        focus_features = extract_features(focus_segments)
        blink_features = extract_features(blink_segments)
        
        # Create labels (0=Relax, 1=Focus, 2=Blink)
        relax_labels = np.zeros(len(relax_features))
        focus_labels = np.ones(len(focus_features))
        blink_labels = np.full(len(blink_features), 2)
        
        # Combine subject data
        subject_features = np.vstack([relax_features, focus_features, blink_features])
        subject_labels = np.hstack([relax_labels, focus_labels, blink_labels])
        subject_ids = [subject_id] * len(subject_labels)
        
        all_features.append(subject_features)
        all_labels.append(subject_labels)
        all_subjects.extend(subject_ids)
        
        print(f" - {subject_id}: Successfully loaded {len(relax_segments)} Relax, {len(focus_segments)} Focus, {len(blink_segments)} Blink segments")
    
    if not all_features:
        return None, None, None
    
    return np.vstack(all_features), np.hstack(all_labels), all_subjects


class EnhancedBCIClassifier:
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=Config.HIDDEN_LAYERS,
            max_iter=Config.MAX_ITER,
            learning_rate_init=Config.LEARNING_RATE,
            alpha=Config.ALPHA,
            activation=Config.ACTIVATION,
            solver=Config.SOLVER,
            batch_size=Config.BATCH_SIZE,
            early_stopping=Config.EARLY_STOPPING,
            validation_fraction=Config.VALIDATION_FRACTION,
            n_iter_no_change=Config.N_ITER_NO_CHANGE,
            random_state=Config.RANDOM_STATE,
            verbose=False
        )
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=Config.N_FEATURES_SELECT) if Config.FEATURE_SELECTION else None
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        if self.feature_selector is not None:
            # Ensure k is not greater than the total number of features
            self.feature_selector.k = min(Config.N_FEATURES_SELECT, X_scaled.shape[1])
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
        else:
            X_selected = X_scaled
        
        self.model.fit(X_selected, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        # ==========================================
        # === STUDENT POSTPROCESSING HERE ========
        # ==========================================
        # Hint: You can use self.model.predict_proba(X_selected) to get probabilities
        # and set custom decision thresholds instead of just using predict().
            
        return self.model.predict(X_selected)
    
    def get_loss_curve(self):
        return self.model.loss_curve_ if hasattr(self.model, 'loss_curve_') else []
    
    def save_model(self,filename='enhanced_bci_classifier.pkl'):
        model_data={
            'model':self.model,
            'scaler':self.scaler,
            'feature_selector':self.feature_selector
        }
        joblib.dump(model_data,filename)
        print(f"\nModel has saved as {filename}")


def leave_one_subject_out_validation():
    print("\nStarting Leave-One-Subject-Out (LOSO) Cross-Validation...")
    
    X, y, subjects = load_all_subjects()
    if X is None: return None
    
    unique_subjects = sorted(list(set(subjects)))
    results = {'accuracies': [], 'confusion_matrices': [], 'loss_curves': [], 'subject_names': []}
    
    print("\n" + "="*40)
    for test_subject in unique_subjects:
        train_mask = [s != test_subject for s in subjects]
        test_mask = [s == test_subject for s in subjects]
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        print(f"Training Model (Test Subject: {test_subject}) | Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        classifier = EnhancedBCIClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        
        results['accuracies'].append(accuracy)
        results['confusion_matrices'].append(cm)
        results['loss_curves'].append(classifier.get_loss_curve())
        results['subject_names'].append(test_subject)
        
        print(f" -> Accuracy: {accuracy:.3f}")

    classifier.save_model('enhanced_bci_classifier.pkl')
    
    return results


def plot_results(results):
    if results is None: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('BCI Classifier (Raw Data) - Group LOSO Results', fontsize=16)
    
    # 1. Accuracy distribution
    subject_names = results['subject_names']
    axes[0].bar(subject_names, results['accuracies'], 
                color=['green' if acc >= 0.7 else 'orange' if acc >= 0.65 else 'red' for acc in results['accuracies']])
    axes[0].set_title('Accuracy by Subject')
    axes[0].set_ylabel('Accuracy')
    axes[0].axhline(y=np.mean(results['accuracies']), color='r', linestyle='--', label=f'Mean: {np.mean(results["accuracies"]):.3f}')
    axes[0].axhline(y=0.65, color='blue', linestyle=':', label='Target: 0.65')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # 2. Overall confusion matrix
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Relax', 'Focus', 'Blink'], yticklabels=['Relax', 'Focus', 'Blink'], ax=axes[1])
    axes[1].set_title('Overall Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    # 3. Training loss curves
    valid_loss_curves = [lc for lc in results['loss_curves'] if len(lc) > 0]
    if valid_loss_curves:
        for i, loss_curve in enumerate(valid_loss_curves):
            axes[2].plot(loss_curve, alpha=0.7, label=subject_names[i])
        axes[2].set_title('Training Loss Curves')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bci_results_raw_data.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("BCI EEG Classification - Group Evaluation")
    print("=" * 60)
    
    results = leave_one_subject_out_validation()
    if results is None:
        print("Validation failed! Please check your directory structure.")
        return
    
    mean_accuracy = np.mean(results['accuracies'])
    std_accuracy = np.std(results['accuracies'])
    
    print("\n" + "="*40)
    print(f"Overall Mean Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        relax_accuracy = total_cm[0, 0] / np.sum(total_cm[0, :]) if np.sum(total_cm[0, :]) > 0 else 0
        concentration_accuracy = total_cm[1, 1] / np.sum(total_cm[1, :]) if np.sum(total_cm[1, :]) > 0 else 0
        blink_accuracy = total_cm[2, 2] / np.sum(total_cm[2, :]) if np.sum(total_cm[2, :]) > 0 else 0
        relax_precision = total_cm[0, 0] / np.sum(total_cm[:, 0]) if np.sum(total_cm[:, 0]) > 0 else 0
        concentration_precision = total_cm[1, 1] / np.sum(total_cm[:, 1]) if np.sum(total_cm[:, 1]) > 0 else 0
        blink_precision = total_cm[2, 2] / np.sum(total_cm[:, 2]) if np.sum(total_cm[:, 2]) > 0 else 0

    print(f"\n[Relax Class]")
    print(f"  - Accuracy (Recall): {relax_accuracy:.3f} ({total_cm[0, 0]}/{np.sum(total_cm[0, :])})")
    print(f"  - Precision: {relax_precision:.3f} ({total_cm[0, 0]}/{np.sum(total_cm[:, 0])})")
    
    print(f"\n[Focus Class]")
    print(f"  - Accuracy (Recall): {concentration_accuracy:.3f} ({total_cm[1, 1]}/{np.sum(total_cm[1, :])})")
    print(f"  - Precision: {concentration_precision:.3f} ({total_cm[1, 1]}/{np.sum(total_cm[:, 1])})")
    
    print(f"\n[Blink Class]")
    print(f"  - Accuracy (Recall): {blink_accuracy:.3f} ({total_cm[2, 2]}/{np.sum(total_cm[2, :])})")
    print(f"  - Precision: {blink_precision:.3f} ({total_cm[2, 2]}/{np.sum(total_cm[:, 2])})")
    
    plot_results(results)
    print(f"\nResults saved to 'bci_results_raw_data.png'")

if __name__ == "__main__":
    main()