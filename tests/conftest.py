"""Put src/ on sys.path so the tests can import hlab and the scripts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
