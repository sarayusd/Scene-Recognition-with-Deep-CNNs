from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "student_model.pth"

DEFAULT_IMAGE_SIZE = 224
TOP_K_PREDICTIONS = 5

SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]