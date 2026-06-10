from chess_assistant.camera import capture_image
def main() -> None:
    image_path = capture_image()
    print(f"Saved image to: {image_path}")

if __name__ == "__main__":
    main()