# Fire and Smoke Detection System

This project implements a real-time fire and smoke detection system using YOLOv8 and Streamlit. The system can process video files and detect instances of fire and smoke, displaying bounding boxes and recording the first detection times.

## Features

- Real-time fire and smoke detection using YOLOv8
- Video file processing with bounding box visualization
- First detection time tracking for both fire and smoke
- User-friendly Streamlit interface
- Support for multiple video formats (mp4, mov, avi, mkv)

## Requirements

- Python 3.8+
- OpenCV
- Streamlit
- Ultralytics (YOLOv8)
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Tofiq055/fireandsmokededection
cd fireandsmokededection
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure the `best.pt` model file is in the project root directory
2. Run the Streamlit app:
```bash
streamlit run app.py
```
3. Open your web browser and navigate to the provided local URL
4. Upload a video file using the interface
5. The system will process the video and display detections in real-time

## Project Structure

- `app.py`: Main application file
- `best.pt`: YOLOv8 model weights
- `videos/`: Directory containing sample videos
- `requirements.txt`: Project dependencies
- `.gitignore`: Git ignore rules

## Model

The project uses a YOLOv8 model trained for fire and smoke detection. The model file (`best.pt`) is included in the repository.

## License

None

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
