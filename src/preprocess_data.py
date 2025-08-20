import pandas as pd
import numpy as np
from PIL import Image
import os
import gc
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
import json
import sys

def setup_logger():
    """Set up comprehensive logging for preprocessing"""
    
    # Create logs directory if it doesn't exist
    log_dir = '../logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create timestamp for unique log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'preprocessing_{timestamp}.log')
    
    # Get logger
    logger = logging.getLogger('DeepfakePreprocessing')
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(logging.DEBUG)
    
    # File handler - logs everything
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler - only INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set formatters
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {os.path.abspath(log_file)}")
    return logger

class MemoryEfficientDataLoader:
    def __init__(self, metadata_path, images_dir, img_size=(224, 224)):
        self.metadata_path = os.path.abspath(metadata_path)
        self.images_dir = os.path.abspath(images_dir)
        self.img_size = img_size
        self.metadata = None
        self.logger = logging.getLogger('DeepfakePreprocessing')
        
    def load_metadata(self):
        """Load and process metadata CSV"""
        self.logger.info(f"Loading metadata from: {self.metadata_path}")
        
        if not os.path.exists(self.metadata_path):
            self.logger.error(f"Metadata file not found: {self.metadata_path}")
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        try:
            self.metadata = pd.read_csv(self.metadata_path)
            self.logger.debug(f"Loaded metadata shape: {self.metadata.shape}")
            
            # Check required columns
            required_cols = ['videoname', 'label']
            missing_cols = [col for col in required_cols if col not in self.metadata.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert video names to image names
            self.metadata['image_name'] = self.metadata['videoname'].str.replace('.mp4', '.jpg')
            self.metadata['label_binary'] = (self.metadata['label'] == 'REAL').astype(int)
            
            # Log statistics
            total = len(self.metadata)
            real = sum(self.metadata['label_binary'])
            fake = total - real
            
            self.logger.info(f"Metadata processed:")
            self.logger.info(f"  Total samples: {total}")
            self.logger.info(f"  REAL: {real} ({real/total*100:.1f}%)")
            self.logger.info(f"  FAKE: {fake} ({fake/total*100:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"Error loading metadata: {str(e)}")
            raise
        
        return self.metadata
    
    def verify_image_files(self):
        """Check which images exist and are valid"""
        self.logger.info(f"Verifying images in: {self.images_dir}")
        
        if not os.path.exists(self.images_dir):
            self.logger.error(f"Images directory not found: {self.images_dir}")
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        existing_files = []
        missing_files = []
        corrupted_files = []
        
        total_files = len(self.metadata)
        
        for idx, row in self.metadata.iterrows():
            img_path = os.path.join(self.images_dir, row['image_name'])
            
            if not os.path.exists(img_path):
                missing_files.append(row['image_name'])
                continue
                
            # Test if image can be opened (but don't use verify())
            try:
                with Image.open(img_path) as img:
                    # Just try to get basic info without corrupting the image
                    _ = img.size
                    _ = img.mode
                existing_files.append(idx)
            except Exception as e:
                corrupted_files.append(row['image_name'])
                self.logger.warning(f"Corrupted image {row['image_name']}: {str(e)}")
        
        self.logger.info(f"Image verification results:")
        self.logger.info(f"  ✓ Valid: {len(existing_files)}/{total_files}")
        self.logger.warning(f"  ✗ Missing: {len(missing_files)}")
        self.logger.error(f"  ⚠ Corrupted: {len(corrupted_files)}")
        
        if len(existing_files) == 0:
            raise ValueError("No valid images found!")
        
        # Filter metadata
        self.metadata = self.metadata.iloc[existing_files].reset_index(drop=True)
        self.logger.info(f"Dataset filtered to {len(self.metadata)} valid samples")
        
        return self.metadata
    
    def process_in_chunks(self, chunk_size=300):
        """Process dataset in chunks"""
        total_samples = len(self.metadata)
        num_chunks = (total_samples - 1) // chunk_size + 1
        
        self.logger.info(f"Processing {total_samples} samples in {num_chunks} chunks of {chunk_size}")
        
        # Ensure chunks directory exists
        chunks_dir = os.path.abspath('../chunks')
        if not os.path.exists(chunks_dir):
            os.makedirs(chunks_dir)
            self.logger.info(f"Created chunks directory: {chunks_dir}")
        
        all_chunk_files = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk_metadata = self.metadata.iloc[start_idx:end_idx]
            
            self.logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_metadata)} samples)")
            
            try:
                X_chunk, y_chunk = self._load_chunk_safely(chunk_metadata)
                
                if len(X_chunk) > 0:
                    chunk_file = os.path.join(chunks_dir, f'chunk_{chunk_idx}.npz')
                    np.savez_compressed(chunk_file, X=X_chunk, y=y_chunk)
                    all_chunk_files.append(chunk_file)
                    
                    self.logger.info(f"✓ Saved chunk {chunk_idx + 1}: {len(X_chunk)} samples")
                    
                    # Memory cleanup
                    del X_chunk, y_chunk
                    collected = gc.collect()
                    self.logger.debug(f"Memory freed: {collected} objects")
                else:
                    self.logger.warning(f"Chunk {chunk_idx + 1} is empty")
                    
            except Exception as e:
                self.logger.error(f"Failed to process chunk {chunk_idx + 1}: {str(e)}")
                continue
        
        self.logger.info(f"Created {len(all_chunk_files)} chunk files")
        return all_chunk_files
    
    def _load_chunk_safely(self, chunk_metadata):
        """Load chunk with proper error handling"""
        images = []
        labels = []
        failed_count = 0
        
        for idx, row in chunk_metadata.iterrows():
            img_path = os.path.join(self.images_dir, row['image_name'])
            
            try:
                # Load and process image
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize(self.img_size, Image.Resampling.LANCZOS)
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                
                images.append(img_array)
                labels.append(row['label_binary'])
                
                # Clean up immediately
                img.close()
                del img, img_resized, img_array
                
            except Exception as e:
                failed_count += 1
                self.logger.debug(f"Failed to load {img_path}: {str(e)}")
                continue
        
        if failed_count > 0:
            self.logger.warning(f"Failed to load {failed_count} images in chunk")
        
        # Convert to numpy arrays if we have data
        if images:
            return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)
        else:
            return np.array([]), np.array([])
    
    def create_data_splits(self, chunk_files):
        """Create train/val/test splits from chunks"""
        self.logger.info("Creating data splits...")
        
        # Load all labels for stratification
        all_labels = []
        chunk_sizes = []
        
        for chunk_file in chunk_files:
            try:
                with np.load(chunk_file) as data:
                    labels = data['y']
                    all_labels.extend(labels)
                    chunk_sizes.append(len(labels))
                    
            except Exception as e:
                self.logger.error(f"Error reading {chunk_file}: {str(e)}")
                continue
        
        all_labels = np.array(all_labels)
        total = len(all_labels)
        
        if total == 0:
            raise ValueError("No labels found in chunks!")
        
        self.logger.info(f"Total samples for splitting: {total}")
        
        # Create indices
        indices = np.arange(total)
        
        # Stratified split: 70% train, 15% val, 15% test
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.3, random_state=42, stratify=all_labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42, 
            stratify=all_labels[temp_idx]
        )
        
        # Log split info
        splits = {
            'train': (train_idx, 'Train'),
            'val': (val_idx, 'Validation'), 
            'test': (test_idx, 'Test')
        }
        
        for key, (split_idx, name) in splits.items():
            split_labels = all_labels[split_idx]
            real_count = np.sum(split_labels)
            fake_count = len(split_labels) - real_count
            
            self.logger.info(f"{name}: {len(split_idx)} samples ({len(split_idx)/total*100:.1f}%)")
            self.logger.info(f"  REAL: {real_count}, FAKE: {fake_count}")
        
        # Save split information
        split_info = {
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist(),
            'test_indices': test_idx.tolist(),
            'total_samples': total,
            'chunk_files': chunk_files,
            'chunk_sizes': chunk_sizes,
            'created_at': datetime.now().isoformat(),
            'image_size': self.img_size
        }
        
        # Ensure processed directory exists
        processed_dir = os.path.abspath('../data/processed')
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        
        split_file = os.path.join(processed_dir, 'split_info.json')
        
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        self.logger.info(f"✓ Split info saved: {split_file}")
        return split_info

def main():
    # Setup logging
    logger = setup_logger()
    
    logger.info("🚀 Starting Deepfake Data Preprocessing")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Check if we're in the right directory
        current_dir = os.getcwd()
        logger.info(f"Working directory: {current_dir}")
        
        # Initialize loader with correct paths
        loader = MemoryEfficientDataLoader(
            metadata_path='../data/metadata.csv',
            images_dir='../data/faces_224',
            img_size=(224, 224)
        )
        
        # Step 1: Load metadata
        logger.info("\n📋 Step 1: Loading metadata...")
        metadata = loader.load_metadata()
        
        # Step 2: Verify images
        logger.info("\n🔍 Step 2: Verifying images...")
        metadata = loader.verify_image_files()
        
        # Step 3: Process chunks
        logger.info("\n📦 Step 3: Processing chunks...")
        chunk_files = loader.process_in_chunks(chunk_size=200)  # Smaller chunks
        
        if not chunk_files:
            raise ValueError("No chunks were created!")
        
        # Step 4: Create splits
        logger.info("\n🔀 Step 4: Creating splits...")
        split_info = loader.create_data_splits(chunk_files)
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"\n🎉 Preprocessing completed!")
        logger.info(f"Duration: {duration}")
        logger.info(f"Chunks: {len(chunk_files)}")
        logger.info(f"Ready for training!")
        
    except Exception as e:
        logger.critical(f"❌ Critical error: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()