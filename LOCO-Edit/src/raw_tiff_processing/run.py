from . import extract_frames, split


def main():
    #stage 1 and 2 are fused, the dtype/axes fix happens as each raw tiff is read for extraction
    extract_frames.main()
    print()
    split.main()


if __name__ == "__main__":
    main()
