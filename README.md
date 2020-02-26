# VisPose
VisPose is a tool for annotating 6D object poses and 2D bounding boxes in monocular image sequences. The tool provides an interface for projecting models into an image sequence and tweaking their fit before batch exporting COCO style annotations for the entire sequence.

## Dependencies
### Python
pip install pyopengl glfw pyassimp==4.1.3 transforms3d numpy Pillow pyyaml

Optional:
pip install pycuda

## Examples
Fitting model into image sequence with fine-grained mouse control, using the vispose.py annotation tool

![Output sample](https://github.com/gidobot/gifs/raw/master/VisPose_FittingModel.gif)

Reviewer annotations with reviewer.py

![Output sample](https://github.com/gidobot/gifs/raw/master/VisPose_Reviewer.gif)
