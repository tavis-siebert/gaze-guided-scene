import os
import shutil
from pathlib import Path
from tqdm import tqdm
import dropbox
from typing import Optional, List
import subprocess
from transformers import CLIPProcessor, CLIPModel
from logger import get_logger
from config.config_utils import DotDict

# Initialize logger for this module
logger = get_logger(__name__)

def download_from_dropbox(dbx: dropbox.Dropbox, shared_link: str, target_path: Path) -> None:
    try:
        _, response = dbx.sharing_get_shared_link_file(url=shared_link)
        total_size = int(response.headers.get('Content-Length', 0))

        with open(target_path, 'wb') as f:
            if total_size:
                # Use progress bar if we know the size
                with tqdm(
                    desc=target_path.name,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for data in response.iter_content(1024 * 1024):
                        size = f.write(data)
                        pbar.update(size)
            else:
                # Fallback for when content length is unknown
                logger.info(f"Downloading {target_path.name}...")
                for data in response.iter_content(1024 * 1024):
                    f.write(data)
    except Exception as e:
        if target_path.exists():
            target_path.unlink()  # Clean up partial download
        raise Exception(f"Failed to download from Dropbox: {str(e)}")

def download_raw_videos(dbx: dropbox.Dropbox, links_file_path: Path, raw_videos_dir: Path) -> None:
    """Download raw videos from links in the provided text file."""
    with open(links_file_path, 'r') as f:
        video_links = [line.strip() for line in f if line.strip()]

    for link in tqdm(video_links, desc="Downloading raw videos"):
        # Get clean filename without the ?dl=0 suffix
        video_name = Path(link).name.split('?')[0]
        target_path = raw_videos_dir / video_name
        
        if target_path.exists():
            logger.info(f"Skipping {video_name} - already exists")
            continue
            
        try:
            download_from_dropbox(dbx, link, target_path)
        except Exception as e:
            logger.error(f"Failed to download {video_name}: {e}")

class ScratchDirectories:
    def __init__(self, config: DotDict):
        """
        Initialize directory paths from configuration.
        
        Args:
            config: Configuration dictionary with paths
        """
        # Convert all path strings to Path objects
        self.scratch_dir = Path(config.base.scratch_dir)
        self.scratch_egtea_dir = Path(config.directories.scratch.egtea)
        self.raw_videos_dir = Path(config.dataset.egtea.raw_videos)
        self.cropped_videos_dir = Path(config.dataset.egtea.cropped_videos)
        self.tmp_dir = Path(config.directories.scratch.tmp)
        self.ego_topo_dir = Path(config.directories.scratch.ego_topo)
        self.clip_model_dir = Path(config.models.clip.model_dir)
        
        # URLs
        self.cropped_clips_url = config.external.urls.dropbox_cropped_clips
        self.video_links_url = config.external.urls.dropbox_video_links
        self.ego_topo_repo_url = config.external.urls.ego_topo_repo
        self.clip_model_id = config.models.clip.model_id

    def create_all(self) -> None:
        """Create all required directories."""
        for directory in [self.scratch_egtea_dir, self.raw_videos_dir, self.cropped_videos_dir, self.tmp_dir, self.clip_model_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")

    def is_cropped_videos_empty(self) -> bool:
        """Check if the cropped videos directory is empty."""
        return not any(self.cropped_videos_dir.iterdir()) if self.cropped_videos_dir.exists() else True

def setup_cropped_videos(dbx: dropbox.Dropbox, directories: ScratchDirectories) -> None:
    """Download and extract cropped video clips."""
    if not directories.is_cropped_videos_empty():
        logger.info("Cropped videos directory not empty, skipping download...")
        return

    cropped_clips_tar_path = directories.tmp_dir / "video_clips.tar"

    logger.info("Downloading cropped video clips...")
    download_from_dropbox(dbx, directories.cropped_clips_url, cropped_clips_tar_path)

    logger.info("Extracting cropped video clips...")
    temp_dir = directories.tmp_dir / "extract"
    # Extract to temp directory
    shutil.unpack_archive(cropped_clips_tar_path, temp_dir)
    # Move contents directly to target
    for file_path in (temp_dir / "cropped_clips").iterdir():
        shutil.move(str(file_path), str(directories.cropped_videos_dir / file_path.name))
    
    logger.info("Cleaning up temporary files...")
    cropped_clips_tar_path.unlink()
    shutil.rmtree(temp_dir)

def setup_raw_videos(dbx: dropbox.Dropbox, directories: ScratchDirectories) -> None:
    """Download raw videos using links from the video_links.txt file."""
    video_links_path = directories.tmp_dir / "video_links.txt"

    logger.info("Downloading video links file...")
    download_from_dropbox(dbx, directories.video_links_url, video_links_path)

    logger.info("Downloading raw videos...")
    download_raw_videos(dbx, video_links_path, directories.raw_videos_dir)

    logger.info("Cleaning up temporary files...")
    video_links_path.unlink()

def setup_ego_topo(directories: ScratchDirectories) -> None:
    """Clone ego-topo repository and download train/val splits."""
    if not directories.ego_topo_dir.exists():
        logger.info("Cloning ego-topo repository...")
        subprocess.run(
            ["git", "clone", directories.ego_topo_repo_url, str(directories.ego_topo_dir)],
            check=True
        )
    else:
        logger.info("ego-topo repository already exists, skipping clone...")

    # Make the download script executable and run it from the correct directory
    download_script = Path("scripts/download_splits.sh")
    if not download_script.exists():
        logger.error("download_splits.sh script not found in scripts directory")
        raise FileNotFoundError("download_splits.sh script not found in scripts directory")

    subprocess.run(["chmod", "+x", str(download_script)], check=True)
    
    # Create data directory in ego-topo if it doesn't exist
    data_dir = directories.ego_topo_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    logger.info("Downloading train/val splits...")
    subprocess.run(
        [str(download_script.absolute())],
        cwd=str(directories.ego_topo_dir),
        check=True
    )

def setup_clip_model(directories: ScratchDirectories) -> None:
    """Download CLIP model and processor for offline use."""
    model_dir = directories.clip_model_dir
    model_dir.mkdir(exist_ok=True)
    
    logger.info("Downloading CLIP model and processor...")
    
    # Download and save model
    model = CLIPModel.from_pretrained(directories.clip_model_id)
    processor = CLIPProcessor.from_pretrained(directories.clip_model_id)
    
    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    logger.info(f"CLIP model and processor saved to {model_dir}")

def setup_scratch(config: DotDict, access_token: Optional[str] = None) -> None:
    """Setup the scratch directory for the Egtea Gaze dataset."""
    if not access_token:
        raise ValueError("Dropbox access token is required")

    directories = ScratchDirectories(config)
    directories.create_all()
    
    dbx = dropbox.Dropbox(access_token)
    
    # Step 1: Setup cropped videos
    setup_cropped_videos(dbx, directories)
    
    # Step 2: Setup raw videos
    setup_raw_videos(dbx, directories)
    
    # Step 3: Setup ego-topo repository and splits
    setup_ego_topo(directories)
    
    # Step 4: Download CLIP model for offline use
    setup_clip_model(directories)
    
    logger.info("Setup complete!")