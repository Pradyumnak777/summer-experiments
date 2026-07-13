import os
import debugpy
import torch


if os.getenv("DEBUGPY", "0") == "1":
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach on 5678...")
    debugpy.wait_for_client()


if __name__ == "__main__":
    '''
    load custom diffusion model for running
    '''
    