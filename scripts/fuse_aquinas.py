import mlx_lm.fuse as fuse_module
import os
import sys
from pathlib import Path

# Allow this script to use the backend's canonical model identity when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_identity import MODEL_BASE_ID, MODEL_PATH

ADAPTER_PATH = "models/aquinas_adapters"
SAVE_PATH = MODEL_PATH

print(f"--- Fusing {ADAPTER_PATH} into {MODEL_BASE_ID} ---")

if not os.path.exists(ADAPTER_PATH):
    print(f"--- ERROR: {ADAPTER_PATH} not found. Check your folder structure! ---")
else:
    try:
        # Check for the modern function name first
        if hasattr(fuse_module, "fuse_model"):
            fuse_module.fuse_model(
                model=MODEL_BASE_ID,
                adapter_path=ADAPTER_PATH,
                save_path=SAVE_PATH
            )
        # Fallback to the legacy function name
        elif hasattr(fuse_module, "fuse"):
            fuse_module.fuse(
                model=MODEL_BASE_ID,
                adapter_path=ADAPTER_PATH,
                save_path=SAVE_PATH
            )
        else:
            print("--- ERROR: Could not find fuse function in mlx_lm.fuse ---")
            
        if os.path.exists(SAVE_PATH):
            print(f"--- SUCCESS: Fused model saved to {SAVE_PATH} ---")
            
    except Exception as e:
        print(f"--- ERROR: {e} ---")
