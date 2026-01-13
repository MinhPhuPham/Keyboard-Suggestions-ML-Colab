"""
Data Download and Management for Google Colab

This script handles automated dataset downloading with Google Drive integration
and email notifications for completion.
"""

import os
import urllib.request
import zipfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import requests
from tqdm import tqdm


# Google Drive folder ID for data storage
DRIVE_DATA_FOLDER_ID = "193OgoZeURCIQE_ZmhX37r8FbpBFA8HOy"  # Phu's Data development
NOTIFICATION_EMAIL = "phamminhphueur@gmail.com"


def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """
    Mount Google Drive in Colab.
    
    Args:
        mount_point: Path to mount Drive
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from google.colab import drive
        drive.mount(mount_point, force_remount=True)
        print(f"✓ Google Drive mounted at {mount_point}")
        return True
    except Exception as e:
        print(f"✗ Failed to mount Google Drive: {e}")
        return False


def check_drive_data_exists(drive_path: str) -> bool:
    """
    Check if data already exists in Google Drive.
    
    Args:
        drive_path: Path to check in Drive
        
    Returns:
        True if data exists, False otherwise
    """
    if os.path.exists(drive_path) and os.path.isdir(drive_path):
        files = os.listdir(drive_path)
        if len(files) > 0:
            print(f"✓ Data found in Drive: {drive_path} ({len(files)} files)")
            return True
    
    print(f"✗ No data found in Drive: {drive_path}")
    return False


def download_file(url: str, output_path: str, description: str = "Downloading") -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save file
        description: Description for progress bar
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading from {url}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✓ Downloaded: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def download_swiftkey_corpus(output_dir: str) -> bool:
    """
    Download SwiftKey corpus from Kaggle.
    
    Args:
        output_dir: Directory to save dataset
        
    Returns:
        True if successful, False otherwise
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Try Kaggle download (requires Kaggle API setup)
    print("Attempting to download SwiftKey corpus from Kaggle...")
    
    try:
        # Check if Kaggle API is configured
        kaggle_json = "/root/.kaggle/kaggle.json"
        if not os.path.exists(kaggle_json):
            print("⚠ Kaggle API not configured. Please set up Kaggle credentials.")
            print("Instructions:")
            print("1. Go to https://www.kaggle.com/settings")
            print("2. Create new API token")
            print("3. Upload kaggle.json to Colab")
            return False
        
        # Download using Kaggle API
        import subprocess
        result = subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "therohk/tweets-blogs-news-swiftkey-dataset-4million",
            "-p", output_dir,
            "--unzip"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ SwiftKey corpus downloaded to {output_dir}")
            return True
        else:
            print(f"✗ Kaggle download failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading SwiftKey corpus: {e}")
        return False


def download_cc100_japanese(output_dir: str, sample_size: str = "10%") -> bool:
    """
    Download CC100 Japanese dataset from Hugging Face.
    
    Args:
        output_dir: Directory to save dataset
        sample_size: Percentage of dataset to download
        
    Returns:
        True if successful, False otherwise
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from datasets import load_dataset
        
        print(f"Downloading CC100 Japanese ({sample_size})...")
        
        # Load dataset
        dataset = load_dataset(
            'cc100',
            lang='ja',
            split=f'train[:{sample_size}]',
            cache_dir=output_dir
        )
        
        # Save to jsonl
        output_file = os.path.join(output_dir, "train.jsonl")
        dataset.to_json(output_file)
        
        print(f"✓ CC100 Japanese downloaded: {output_file}")
        print(f"  Total samples: {len(dataset)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading CC100 Japanese: {e}")
        return False


def download_emoji_dataset(output_dir: str) -> bool:
    """
    Download emoji dataset for augmentation.
    
    Args:
        output_dir: Directory to save dataset
        
    Returns:
        True if successful, False otherwise
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create a simple emoji list
        common_emojis = [
            '😊', '😂', '❤️', '👍', '🎉', '😍', '🔥', '✨', '💯', '🙏',
            '😭', '🥰', '😎', '🤔', '👏', '🙌', '💪', '🎊', '🌟', '💖',
            '😅', '🤗', '😇', '🥳', '😘', '💕', '🎈', '🌈', '⭐', '💫'
        ]
        
        output_file = os.path.join(output_dir, "emojis.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            for emoji in common_emojis:
                f.write(f"{emoji}\n")
        
        print(f"✓ Emoji dataset created: {output_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error creating emoji dataset: {e}")
        return False


def send_notification_email(subject: str, message: str, to_email: str = NOTIFICATION_EMAIL) -> bool:
    """
    Send email notification (requires Colab email setup).
    
    Note: This is a placeholder. In Colab, you'll need to configure
    email sending through Gmail API or SMTP with app password.
    
    Args:
        subject: Email subject
        message: Email body
        to_email: Recipient email
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # For Colab, print notification instead
        # In production, configure Gmail API or SMTP
        print("\n" + "="*60)
        print("📧 EMAIL NOTIFICATION")
        print("="*60)
        print(f"To: {to_email}")
        print(f"Subject: {subject}")
        print(f"\n{message}")
        print("="*60 + "\n")
        
        # TODO: Implement actual email sending with Gmail API
        # See: https://developers.google.com/gmail/api/guides/sending
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to send notification: {e}")
        return False


def setup_english_data(drive_base_path: str = "/content/drive/MyDrive/Phu's Data development") -> str:
    """
    Set up English training data with Drive integration.
    
    Args:
        drive_base_path: Base path in Google Drive
        
    Returns:
        Path to English data directory
    """
    data_path = os.path.join(drive_base_path, "data", "english")
    
    print("\n" + "="*60)
    print("ENGLISH DATA SETUP")
    print("="*60)
    
    # Check if data exists
    if check_drive_data_exists(data_path):
        print("✓ Using existing English data from Drive")
        return data_path
    
    # Download data
    print("Downloading English datasets...")
    os.makedirs(data_path, exist_ok=True)
    
    # Download SwiftKey corpus
    swiftkey_success = download_swiftkey_corpus(data_path)
    
    # Download emoji dataset
    emoji_success = download_emoji_dataset(data_path)
    
    if swiftkey_success and emoji_success:
        print("\n✓ English data setup complete!")
        send_notification_email(
            "English Data Download Complete",
            f"English training data has been downloaded to:\n{data_path}\n\nReady for training!"
        )
        return data_path
    else:
        print("\n⚠ English data setup incomplete. Please check errors above.")
        return None


def setup_japanese_data(drive_base_path: str = "/content/drive/MyDrive/Phu's Data development") -> str:
    """
    Set up Japanese training data with Drive integration.
    
    Args:
        drive_base_path: Base path in Google Drive
        
    Returns:
        Path to Japanese data directory
    """
    data_path = os.path.join(drive_base_path, "data", "japanese")
    
    print("\n" + "="*60)
    print("JAPANESE DATA SETUP")
    print("="*60)
    
    # Check if data exists
    if check_drive_data_exists(data_path):
        print("✓ Using existing Japanese data from Drive")
        return data_path
    
    # Download data
    print("Downloading Japanese datasets...")
    os.makedirs(data_path, exist_ok=True)
    
    # Download CC100 Japanese
    cc100_success = download_cc100_japanese(data_path, sample_size="10%")
    
    # Download emoji dataset
    emoji_success = download_emoji_dataset(data_path)
    
    if cc100_success and emoji_success:
        print("\n✓ Japanese data setup complete!")
        send_notification_email(
            "Japanese Data Download Complete",
            f"Japanese training data has been downloaded to:\n{data_path}\n\nReady for training!"
        )
        return data_path
    else:
        print("\n⚠ Japanese data setup incomplete. Please check errors above.")
        return None


if __name__ == "__main__":
    print("Data Download and Management Utilities")
    print("Import this module in your Colab notebooks")
    print("\nExample:")
    print("  from src.colab_data_manager import mount_google_drive, setup_english_data")
    print("  mount_google_drive()")
    print("  data_path = setup_english_data()")
