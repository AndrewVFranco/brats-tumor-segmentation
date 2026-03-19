import synapseclient
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

syn = synapseclient.Synapse()
syn.login(authToken=os.getenv("SYNAPSE_AUTH_TOKEN"))

downloads = syn.get_download_list(downloadLocation=str(DATA_DIR))