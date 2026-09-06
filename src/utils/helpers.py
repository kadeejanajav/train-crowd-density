from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it doesn't exist, return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(folder: str | Path, extensions=(".jpg", ".jpeg", ".png")) -> list[str]:
    folder = Path(folder)
    return [str(f) for f in folder.iterdir() if f.suffix.lower() in extensions]
