import viewer as v
import numpy as np
import glfw # lean window system wrapper for OpenGL
import argparse
import transforms3d as tf3d
import json
import sys
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
    global SAVE_FLAG
    global PLAY
    global BBOX
    global BLACKLIST
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_SPACE:
            PLAY = ~PLAY
        elif key == glfw.KEY_B:
            BBOX = ~BBOX
            UPDATE_FLAG = 1
        elif key == glfw.KEY_S:
            SAVE_FLAG = 1
        elif ~PLAY:
            if key == glfw.KEY_LEFT:
                if INDEX == 0:
                    INDEX = MAX_INDEX
                else:
                    INDEX -= 1
                UPDATE_FLAG = 1
            elif key == glfw.KEY_RIGHT:
                if INDEX == MAX_INDEX:
                    INDEX = 0
                else:
                    INDEX += 1
                UPDATE_FLAG = 1
            elif key == glfw.KEY_M:
                if INDEX in BLACKLIST:
                    BLACKLIST.remove(INDEX)
                else:
                    BLACKLIST.append(INDEX)
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


def saveCulledAnnotations(annotations, outfile):
    global BLACKLIST
    global IMAGE_ID_LIST
    cull_ids = [IMAGE_ID_LIST[i] for i in BLACKLIST]
    cull_annotations = {}
    cull_annotations['images'] = []
    cull_annotations['annotations'] = []
    cull_annotations['categories'] = annotations['categories']
    print("Culling annotations...")
    for i, image in enumerate(annotations['images']):
        if image['id'] not in cull_ids:
            cull_annotations['images'].append(annotations['images'][i])
    for i, ann in enumerate(annotations['annotations']):
        if ann['image_id'] not in cull_ids:
            cull_annotations['annotations'].append(annotations['annotations'][i])
    print("Saving annotations to {}".format(outfile))
    with open(outfile, 'wb') as outfile:
            outfile.write(json.dumps(cull_annotations))
    print("Finished saving annotations")


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
    parser.add_argument('--rate', type=int, default=10,
        help='Framerate for playback.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    global INDEX; INDEX = 0
    global MAX_INDEX; MAX_INDEX = 0
    global UPDATE_FLAG; UPDATE_FLAG = 1
    global SAVE_FLAG; SAVE_FLAG = 0
    global PLAY; PLAY = False
    global BBOX; BBOX = False
    global BLACKLIST; BLACKLIST = []
    global IMAGE_ID_LIST; IMAGE_ID_LIST = []
    CATEGORIES = {}

    annotations = parseAnnotations(args.ann)
    MAX_INDEX = len(annotations['images']) - 1
    IMAGE_ID_LIST = [image['id'] for image in annotations['images']]
    CATEGORIES = {}
    for c in annotations['categories']:
        CATEGORIES[c['id']] = c['name']
    images = refactorAnnotations(annotations)

    camera = v.Camera(args.camera)
    viewer = v.Viewer(camera, background=osp.join(args.images, images[IMAGE_ID_LIST[0]]['file_name']))
    models = parseModels(args.models)

    for model in models:
        print(model['name'])
        viewer.add(*[mesh for mesh in v.load_textured(model['path'], model['name'])])

    glfw.set_key_callback(viewer.win, keyCallback)

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
            viewer.clear_markers()
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
                if INDEX in BLACKLIST:
                    viewer.add_marker(center=[0.9,0.9], width=0.1, color=[1,0.5,0])
            viewer.render()
            sys.stdout.write('image: {}/{}  -  b -> toggle boxes, m -> mark frame, s -> cull and save, <space> -> toggle play, <left, right> -> move single frame     \r'.format(INDEX, MAX_INDEX))
            sys.stdout.flush()
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

        if SAVE_FLAG:
            SAVE_FLAG = 0
            outfile = osp.splitext(args.ann)[0] + '_culled' + osp.splitext(args.ann)[1]
            saveCulledAnnotations(annotations, outfile)
