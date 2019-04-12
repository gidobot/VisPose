import viewer as v
import numpy as np
import glfw # lean window system wrapper for OpenGL
import argparse
import transforms3d as tf3d
import json
from os import path as osp
from time import time


def parseAnnotations(file):
    with open(file, 'r') as f:
        annotations = json.load(f) 
    return annotations

def parseModels(file):
    models = []
    with open(file, 'r') as f:
        for line in f:
            model = {}
            vals = [x for x in line.strip().split(' ')]
            model['name'] = vals[0]
            model['path'] = vals[1]
            models.append(model)
    return models

def keyCallback(win, key, scancode, action, mods):
    global INDEX
    global MAX_INDEX
    global UPDATE_FLAG
    global PLAY
    global BBOX
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_SPACE:
            PLAY = ~PLAY
        elif key == glfw.KEY_B:
            BBOX = ~BBOX
            UPDATE_FLAG = 1
        elif ~PLAY:
            if key == glfw.KEY_LEFT:
                if INDEX != 0:
                    INDEX -= 1
                    UPDATE_FLAG = 1
            elif key == glfw.KEY_RIGHT:
                if INDEX != MAX_INDEX:
                    INDEX += 1
                    UPDATE_FLAG = 1

def refactorAnnotations(annotations):
    images = {}
    for image in annotations['images']:
        image_id = image['id']
        images[image_id] = {}
        images[image_id]['file_name'] = image['file_name']
        images[image_id]['annotations'] = []
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        images[image_id]['annotations'].append(ann)
    return images

def parse_args():
    parser = argparse.ArgumentParser(description='Tool for annotating monocular image sequences with groundtruth poses.')
    parser.add_argument('--camera', type=str, required=True,
        help='Camera calibration yaml file in camchain format. Expected camera name is cam0.')
    parser.add_argument('--images', type=str, required=True,
        help='Folder containing image sequence. Expected naming convention is consecutive number sequence starting from 0.png.')
    parser.add_argument('--ann', type=str, required=True,
        help='JSON file containing pose annotations for image sequence. Expected format is COCO style as exported from VisPose.')
    parser.add_argument('--models', type=str, required=True,
        help='Text file containing model names and paths.')
    parser.add_argument('--rate', type=int, default=5,
        help='Framerate for playback.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    global INDEX; INDEX = 0
    global MAX_INDEX; MAX_INDEX = 0
    global UPDATE_FLAG; UPDATE_FLAG = 1
    global PLAY; PLAY = False
    global BBOX; BBOX = False
    IMAGE_ID_LIST = []
    CATEGORIES = {}

    camera = v.Camera(args.camera)
    viewer = v.Viewer(camera, background=osp.join(args.images, '0.png'))
    annotations = parseAnnotations(args.ann)
    models = parseModels(args.models)

    for model in models:
        print(model['name'])
        viewer.add(*[mesh for mesh in v.load_textured(model['path'], model['name'])])

    glfw.set_key_callback(viewer.win, keyCallback)

    MAX_INDEX = len(annotations['images']) - 1
    IMAGE_ID_LIST = [image['id'] for image in annotations['images']]
    CATEGORIES = {}
    for c in annotations['categories']:
        CATEGORIES[c['id']] = c['name']
    images = refactorAnnotations(annotations)

    # for i, ann in enumerate(annotations['annotations']):
    #     print("{} of {}".format(i,len(annotations['annotations'])))
    #     name = CATEGORIES[ann['category_id']]
    #     viewer.set_active(name)
    #     Rt = np.eye(4)
    #     Rt[0:3,3] = ann['pose'][0:3]
    #     Rt[0:3,0:3] = tf3d.quaternions.quat2mat(ann['pose'][3:])
    #     viewer.set_pose_matrix(Rt)
    #     annotations['annotations'][i] = viewer.bounding_box()
    # with open(args.ann, 'wb') as outfile:
    #         outfile.write(json.dumps(annotations))

    while not glfw.window_should_close(viewer.win):
        now = time()
        if UPDATE_FLAG:
            image = images[IMAGE_ID_LIST[INDEX]]
            viewer.background.set(osp.join(args.images, image['file_name']))
            viewer.clear_bboxes()
            for ann in image['annotations']:
                name = CATEGORIES[ann['category_id']]
                viewer.set_active(name)
                Rt = np.eye(4)
                Rt[0:3,3] = ann['pose'][0:3]
                Rt[0:3,0:3] = tf3d.quaternions.quat2mat(ann['pose'][3:])
                viewer.set_pose_matrix(Rt)
                if BBOX and ann['pose'][2] > 0:
                    viewer.add_bbox(ann['bbox'])
                    # viewer.add_bbox(viewer.bounding_box())
            viewer.render()
            UPDATE_FLAG = 0

        glfw.poll_events()

        if PLAY:
            if INDEX != MAX_INDEX:
                INDEX += 1
            else:
                INDEX = 0
            UPDATE_FLAG = 1
            while 1/float(args.rate) > time()-now:
                pass
