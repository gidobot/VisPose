# VisPose
VisPose is a tool for annotating 6D object poses and 2D bounding boxes in monocular image sequences. The tool provides an interface for projecting models into an image sequence and tweaking their fit before batch exporting COCO style annotations for the entire sequence.

## Dependencies
### Python
pip install pyopengl glfw pyassimp==4.1.3 transforms3d numpy Pillow pyyaml

Optional:
pip install pycuda

## Usage

### Annotating a sequence with VisPose

python vispose.py --camera <path .yaml calibration file> --images <path to folder containing images> --poses <path to camera pose file>

See sample calibration files under the calibration folder for formatting. Currently, the only supported camera model for annotating with VisPose.py is pinhole. Parameters for the pinhole camera are specified under the cam0 tag in the yaml file. Fisheye image sequences can be annotated by first rectifying the image sequence to perspective and processing this rectified sequence through the annotation tool. The exported pose annotations are valid for both the perspective and raw fisheye images. The bounding box annotations can be regenerated for the raw fisheye images using the reviewer tool described below.

Currently, the only supported image format is png, and images are expected to be named as sequential integers (e.g. 0.png, 1.png, 2.png, ...).

The camera pose file is CSV format, containing pose annotations for each image in the sequence. The expected format for each line in file is

image#, x, y, z, qw, qx, qy, qz

where image# is the integer name of the image in the sequence, (x,y,z) is the globally referenced camera translation, and (qw, qx, qy, qz) is the globally referenced camera orientation. The global reference frame is arbitrary and can be obtained by running the image sequence through any compatable SLAM method.

When annotations are exported, they are saved to a .json file one folder above the image folder.

### Reviewing a sequence

## Examples
Fitting model into image sequence with fine-grained mouse control, using the vispose.py annotation tool

![Output sample](https://github.com/gidobot/gifs/raw/master/VisPose_FittingModel.gif)

Reviewing annotations with reviewer.py

![Output sample](https://github.com/gidobot/gifs/raw/master/VisPose_Reviewer.gif)
