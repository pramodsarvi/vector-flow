from huggingface_hub import snapshot_download

for r in ["IDEA-Research/grounding-dino-base", "facebook/sam2.1-hiera-small",
          "google/siglip2-base-patch16-224", "Ruicheng/moge-2-vitl-normal"]:
    print(r, snapshot_download(r), flush=True)
