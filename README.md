# Object-Tracking-from-video

## Project Description
This repository contains code for ENPM673 Project1 - Track an object's trajectory from a video.

![alt text](https://github.com/suhasnagaraj99/Object-Tracking-from-video/blob/main/Results/path.jpg?raw=false)

### Required Libraries
Before running the code, ensure that the following Python libraries are installed:

- `cv2`
- `numpy`
- `matplotlib`

You can install if they are not already installed:

```bash
sudo apt-get install python3-opencv python3-numpy python3-matplotlib
```

### Running the Code
Follow these steps to run the code:

1. Make sure the video `project1_video1.mp4` is pasted in the same directory as the `suhas99_project1.py` file.
2. Execute the `suhas99_project1.py`; run the following command in your terminal:

```bash
python3 suhas99_project1.py
```
3. The script computes the trajectory of the dynamic object (object of interest) in the video by using the concepts of masking and curve fitting. The computed trajectory is saved as `Trajectory.png`.
4. The script also plots the path genereted on a frame of the video, for visualization, and saves it as `path.jpg`
