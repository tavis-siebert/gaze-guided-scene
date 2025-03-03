import os
import shutil
from pathlib import Path
from tqdm import tqdm
import dropbox
from typing import Optional, List
import subprocess

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
                print(f"Downloading {target_path.name}...")
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
            print(f"Skipping {video_name} - already exists")
            continue
            
        try:
            download_from_dropbox(dbx, link, target_path)
        except Exception as e:
            print(f"Failed to download {video_name}: {e}")

class ScratchDirectories:
    def __init__(self, scratch_dir: Path):
        self.scratch_dir = scratch_dir
        self.egtea_dir = scratch_dir / "egtea_gaze"
        self.raw_videos_dir = self.egtea_dir / "raw_videos"
        self.cropped_videos_dir = self.egtea_dir / "cropped_videos"
        self.tmp_dir = scratch_dir / "tmp"
        self.ego_topo_dir = scratch_dir / "ego-topo"

    def create_all(self) -> None:
        for directory in [self.egtea_dir, self.raw_videos_dir, self.cropped_videos_dir, self.tmp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def is_cropped_videos_empty(self) -> bool:
        return not any(self.cropped_videos_dir.iterdir()) if self.cropped_videos_dir.exists() else True

def setup_cropped_videos(dbx: dropbox.Dropbox, directories: ScratchDirectories) -> None:
    """Download and extract cropped video clips."""
    if not directories.is_cropped_videos_empty():
        print("Cropped videos directory not empty, skipping download...")
        return

    cropped_clips_shared_link = "https://www.dropbox.com/scl/fi/97r0kjz65wb6xf0mjpcd0/video_clips.tar"
    cropped_clips_tar_path = directories.tmp_dir / "video_clips.tar"

    print("Downloading cropped video clips...")
    download_from_dropbox(dbx, cropped_clips_shared_link, cropped_clips_tar_path)

    print("Extracting cropped video clips...")
    temp_dir = directories.tmp_dir / "extract"
    # Extract to temp directory
    shutil.unpack_archive(cropped_clips_tar_path, temp_dir)
    # Move contents directly to target
    for file_path in (temp_dir / "cropped_clips").iterdir():
        shutil.move(str(file_path), str(directories.cropped_videos_dir / file_path.name))
    
    print("Cleaning up temporary files...")
    cropped_clips_tar_path.unlink()
    shutil.rmtree(temp_dir)

def setup_raw_videos(dbx: dropbox.Dropbox, directories: ScratchDirectories) -> None:
    """Download raw videos using links from the video_links.txt file."""
    video_links_url = "https://www.dropbox.com/scl/fi/o7mrc7okncgoz14a49e5q/video_links.txt"
    video_links_path = directories.tmp_dir / "video_links.txt"

    print("Downloading video links file...")
    download_from_dropbox(dbx, video_links_url, video_links_path)

    print("Downloading raw videos...")
    download_raw_videos(dbx, video_links_path, directories.raw_videos_dir)

    print("Cleaning up temporary files...")
    video_links_path.unlink()

def setup_ego_topo(directories: ScratchDirectories) -> None:
    """Clone ego-topo repository and download train/val splits."""
    if not directories.ego_topo_dir.exists():
        print("Cloning ego-topo repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/ego-topo.git", str(directories.ego_topo_dir)],
            check=True
        )
    else:
        print("ego-topo repository already exists, skipping clone...")

    # Make the download script executable and run it from the correct directory
    download_script = Path("scripts/download_splits.sh")
    if not download_script.exists():
        raise FileNotFoundError("download_splits.sh script not found in scripts directory")

    subprocess.run(["chmod", "+x", str(download_script)], check=True)
    
    # Create data directory in ego-topo if it doesn't exist
    data_dir = directories.ego_topo_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    print("Downloading train/val splits...")
    subprocess.run(
        [str(download_script.absolute())],
        cwd=str(directories.ego_topo_dir),
        check=True
    )

def setup_scratch(config, access_token: Optional[str] = None) -> None:
    """Setup the scratch directory for the Egtea Gaze dataset."""
    if not access_token:
        raise ValueError("Dropbox access token is required")

    directories = ScratchDirectories(Path(config.paths.scratch_dir))
    directories.create_all()
    
    dbx = dropbox.Dropbox(access_token)
    
    # Step 1: Setup cropped videos
    setup_cropped_videos(dbx, directories)
    
    # Step 2: Setup raw videos
    setup_raw_videos(dbx, directories)
    
    # Step 3: Setup ego-topo repository and splits
    setup_ego_topo(directories)
    
    print("Setup complete!")