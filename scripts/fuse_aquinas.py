import mlx_lm.fuse as fuse_module
import os

# Your refined paths
BASE_MODEL = "google/gemma-4-E2B-it"
ADAPTER_PATH = "models/aquinas_adapters"
SAVE_PATH = "models/Aquinas-Final"

print(f"--- Fusing {ADAPTER_PATH} into {BASE_MODEL} ---")

if not os.path.exists(ADAPTER_PATH):
    print(f"--- ERROR: {ADAPTER_PATH} not found. Check your folder structure! ---")
else:
    try:
        # Check for the modern function name first
        if hasattr(fuse_module, "fuse_model"):
            fuse_module.fuse_model(
                model=BASE_MODEL,
                adapter_path=ADAPTER_PATH,
                save_path=SAVE_PATH
            )
        # Fallback to the legacy function name
        elif hasattr(fuse_module, "fuse"):
            fuse_module.fuse(
                model=BASE_MODEL,
                adapter_path=ADAPTER_PATH,
                save_path=SAVE_PATH
            )
        else:
            print("--- ERROR: Could not find fuse function in mlx_lm.fuse ---")
            
        if os.path.exists(SAVE_PATH):
            print(f"--- SUCCESS: Fused model saved to {SAVE_PATH} ---")
            
    except Exception as e:
        print(f"--- ERROR: {e} ---")
