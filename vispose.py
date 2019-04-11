from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSlot, QTimer
from mainwindow import Ui_MainWindow
import sys, os
from os import path as osp
from threading import Thread
import viewer2 as v
import numpy as np
import glfw # lean window system wrapper for OpenGL
import argparse
import transforms3d as tf3d
from tqdm import tqdm
import json

class MainWindowUIClass( Ui_MainWindow ):
    def __init__(self, camera_cal, image_folder, pose_file, main_window):
        glfw.init()             # initialize window system glfw
        super(MainWindowUIClass, self).__init__()

        # Parse pose annotation file
        self.num_images = 0
        self.image_index = 0
        self.image_folder = image_folder
        self.image_format = '.png'  # TODO: support other formats
        self.pose_annotations = self.parsePoseFile(pose_file)
        self.Rt = np.eye(4)

        """ create a window, add scene objects, then run rendering loop """
        camera = v.Camera(camera_cal)
        self.viewer = v.Viewer(camera, background=osp.join(image_folder, '0.png'))

        # place instances of our basic objects
        # viewer.add(*[mesh for file in sys.argv[1:] for mesh in load(file)])
        # self.viewer.add(*[mesh for mesh in v.load_textured('mug.obj')])
        # viewer.add(*[TexturedPlane("texture_map.png")])

        # Rt =  np.array([[0.9285,   -0.3686,    0.0458,    0.0957],
        #               [-0.1438,   -0.4703,   -0.8707,    0.0042],
        #               [0.3424,    0.8018,   -0.4897,    0.6644]])

        # self.viewer.set_view_from_pose_matrix(Rt)

        self.timer = QTimer()
        self.timer.timeout.connect(self.renderSlot)
        self.timer.start(33)

        # Init gui items
        self.setupUi(main_window)
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(self.num_images-1)

        self.setImageIndex(0)

    def __del__(self):
        glfw.terminate()        # destroy all glfw windows and GL contexts
        self.t.join()
        
    def setupUi( self, MW ):
        ''' Setup the UI of the super class, and add here code
        that relates to the way we want our UI to operate.
        '''
        super(MainWindowUIClass, self).setupUi( MW )

    def refreshAll( self ):
        self.updatePose()
        self.horizontalSlider.setValue(self.image_index)
    
    # slot
    # def returnPressedSlot( self ):
    #     ''' Called when the user enters a string in the line edit and
    #     presses the ENTER key.
    #     '''
    #     fileName =  self.lineEdit.text()
    #     if self.model.isValid( fileName ):
    #         self.model.setFileName( self.lineEdit.text() )
    #         self.refreshAll()
    #     else:
    #         m = QtWidgets.QMessageBox()
    #         m.setText("Invalid file name!\n" + fileName )
    #         m.setIcon(QtWidgets.QMessageBox.Warning)
    #         m.setStandardButtons(QtWidgets.QMessageBox.Ok
    #                              | QtWidgets.QMessageBox.Cancel)
    #         m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
    #         ret = m.exec_()
    #         self.lineEdit.setText( "" )
    #         self.refreshAll()

    def setImageIndex(self, idx):
        if idx < 0:
            idx = 0
        elif idx > self.num_images - 1:
            idx = self.num_images - 1
        self.image_index = idx
        image = self.pose_annotations[idx][0]
        pose = self.pose_annotations[idx][1:]
        self.viewer.background.set(osp.join(self.image_folder,str(image)+self.image_format))
        self.viewer.set_view_from_pose_vec(pose, invert=False)
        self.refreshAll()

    def updatePose(self):
        Rt = self.viewer.get_pose_matrix()
        if not np.allclose(Rt, self.Rt):
            self.Rt = Rt
            t = Rt[0:3,3]
            r = tf3d.euler.mat2euler(Rt[0:3,0:3], axes='sxyz')

            self.lineEdit_3.setText("{:.3f}".format(t[0]))
            self.lineEdit_4.setText("{:.3f}".format(t[1]))
            self.lineEdit_5.setText("{:.3f}".format(t[2]))

            self.lineEdit_6.setText("{:.2f}".format(r[0]*180.0/np.pi))
            self.lineEdit_7.setText("{:.2f}".format(r[1]*180.0/np.pi))
            self.lineEdit_8.setText("{:.2f}".format(r[2]*180.0/np.pi))

    def poseSlot(self):
        Rt = self.viewer.get_pose_matrix()
        t = Rt[0:3,3]
        r = list(tf3d.euler.mat2euler(Rt[0:3,0:3], axes='sxyz'))
        try:
            t[0] = float(self.lineEdit_3.text())
            t[1] = float(self.lineEdit_4.text())
            t[2] = float(self.lineEdit_5.text())
            Rt[0:3,3] = t
            r[0] = float(self.lineEdit_6.text())*np.pi/180
            r[1] = float(self.lineEdit_7.text())*np.pi/180
            r[2] = float(self.lineEdit_8.text())*np.pi/180
            Rt[0:3,0:3] = tf3d.euler.euler2mat(*r, axes='sxyz')
            self.viewer.set_pose_matrix(Rt)
            self.refreshAll()
        except:
            print("Pose fields must be float values")

    def transXNSlot(self):
        Rt = self.viewer.get_pose_matrix()
        t = Rt[0:3,3]
        t[0] -= 0.01
        Rt[0:3,3] = t
        self.viewer.set_pose_matrix(Rt)
        self.refreshAll()

    def transXPSlot(self):
        Rt = self.viewer.get_pose_matrix()
        t = Rt[0:3,3]
        t[0] += 0.01
        Rt[0:3,3] = t
        self.viewer.set_pose_matrix(Rt)
        self.refreshAll()

    def transYNSlot(self):
        Rt = self.viewer.get_pose_matrix()
        t = Rt[0:3,3]
        t[1] -= 0.01
        Rt[0:3,3] = t
        self.viewer.set_pose_matrix(Rt)
        self.refreshAll()

    def transYPSlot(self):
        Rt = self.viewer.get_pose_matrix()
        t = Rt[0:3,3]
        t[1] += 0.01
        Rt[0:3,3] = t
        self.viewer.set_pose_matrix(Rt)
        self.refreshAll()

    def transZNSlot(self):
        Rt = self.viewer.get_pose_matrix()
        t = Rt[0:3,3]
        t[2] -= 0.01
        Rt[0:3,3] = t
        self.viewer.set_pose_matrix(Rt)
        self.refreshAll()

    def transZPSlot(self):
        Rt = self.viewer.get_pose_matrix()
        t = Rt[0:3,3]
        t[2] += 0.01
        Rt[0:3,3] = t
        self.viewer.set_pose_matrix(Rt)
        self.refreshAll()

    def rotRNSlot(self):
        Rt = self.viewer.get_pose_matrix()
        r = list(tf3d.euler.mat2euler(Rt[0:3,0:3], axes='sxyz'))
        r[0] -= 1.0*np.pi/180
        Rt[0:3,0:3] = tf3d.euler.euler2mat(*r, axes='sxyz')
        self.viewer.set_pose_matrix(Rt)
        self.refreshAll()

    def rotRPSlot(self):
        Rt = self.viewer.get_pose_matrix()
        r = list(tf3d.euler.mat2euler(Rt[0:3,0:3], axes='sxyz'))
        r[0] += 1.0*np.pi/180
        Rt[0:3,0:3] = tf3d.euler.euler2mat(*r, axes='sxyz')
        self.viewer.set_pose_matrix(Rt)
        self.refreshAll()

    def rotPNSlot(self):
        Rt = self.viewer.get_pose_matrix()
        r = list(tf3d.euler.mat2euler(Rt[0:3,0:3], axes='sxyz'))
        r[1] -= 1.0*np.pi/180
        Rt[0:3,0:3] = tf3d.euler.euler2mat(*r, axes='sxyz')
        self.viewer.set_pose_matrix(Rt)
        self.refreshAll()

    def rotPPSlot(self):
        Rt = self.viewer.get_pose_matrix()
        r = list(tf3d.euler.mat2euler(Rt[0:3,0:3], axes='sxyz'))
        r[1] += 1.0*np.pi/180
        Rt[0:3,0:3] = tf3d.euler.euler2mat(*r, axes='sxyz')
        self.viewer.set_pose_matrix(Rt)
        self.refreshAll()

    def rotYNSlot(self):
        Rt = self.viewer.get_pose_matrix()
        r = list(tf3d.euler.mat2euler(Rt[0:3,0:3], axes='sxyz'))
        r[2] -= 1.0*np.pi/180
        Rt[0:3,0:3] = tf3d.euler.euler2mat(*r, axes='sxyz')
        self.viewer.set_pose_matrix(Rt)
        self.refreshAll()

    def rotYPSlot(self):
        Rt = self.viewer.get_pose_matrix()
        r = list(tf3d.euler.mat2euler(Rt[0:3,0:3], axes='sxyz'))
        r[2] += 1.0*np.pi/180
        Rt[0:3,0:3] = tf3d.euler.euler2mat(*r, axes='sxyz')
        self.viewer.set_pose_matrix(Rt)
        self.refreshAll()

    def prevImageSlot(self):
        idx = self.image_index - 1
        self.setImageIndex(idx)

    def nextImageSlot(self):
        idx = self.image_index + 1
        self.setImageIndex(idx)

    def imageSliderMovedSlot(self, idx):
        self.setImageIndex(idx)

    def parsePoseFile(self, file):
        poses = []
        with open(file) as fp:
            for cnt, line in enumerate(fp):
                # [img#, x, y, z, qx, qy, qz]
                pose = [float(x) for x in line.strip().split(',')]
                pose[0] = int(pose[0])
                poses.append(pose)
            self.num_images = cnt+1
            self.image_index = 0
        return poses


    def renderSlot(self):
        self.viewer.render()
        self.refreshAll()

    # slot
    def browseSlot( self ):
        ''' Called when the user presses the Browse button
        '''
        #self.debugPrint( "Browse button pressed" )
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "QFileDialog.getOpenFileName()",
                        "",
                        "All Files (*);;Python Files (*.py)",
                        options=options)
        if fileName:
            self.debugPrint( "setting file name: " + fileName )
            self.model.setFileName( fileName )
            self.refreshAll()

    # slot
    def loadModelSlot( self ):
        ''' Called when the user presses the load button.
        '''
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "QFileDialog.getOpenFileName()",
                        "",
                        "All Files (*);;OBJ Files (*.obj *.OBJ)",
                        options=options)
        if fileName:
            basename = os.path.basename(fileName)
            itemsTextList =  self.readNames()
            # Uniquely identify objects with same filename
            basenametmp = basename
            i = 0
            while any(basenametmp == s for s in itemsTextList):
                i += 1
                basenametmp = basename + '-' + str(i)
            basename = basenametmp
            self.viewer.add(*[mesh for mesh in v.load_textured(fileName, basename)])
            self.listWidget.addItems([basename])
            self.listWidget.setCurrentRow(self.listWidget.count()-1)
            # item = self.listWidget.currentItem()
            # item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)

            # self.refreshAll()
        self.refreshAll()

    # slot
    def deleteModelSlot( self ):
        item = self.listWidget.currentItem()
        if not item: return        
        objName = item.text()
        self.viewer.remove(objName)
        self.listWidget.takeItem(self.listWidget.row(item))
        self.refreshAll()

    def editItemSlot(self):
        # TODO: there will be name conflict if rename is same as another object.
        #       need to implement proper handling
        currItem = self.listWidget.currentItem()
        if not currItem: return
        currName = currItem.text()
        text, ok = QtWidgets.QInputDialog.getText(self.centralWidget, 'Class Label Dialog', currName)
        if ok:
            newName = str(text)
            self.viewer.rename(currName, newName)
            currItem.setText(newName)

    def readNames(self):
        return [str(self.listWidget.item(i).text()) for i in range(self.listWidget.count())]

    def exportSlot(self):
        ann_dict = {}
        images = []
        annotations = []
        json_name = 'annotations.json'
        ann_id = 0
        image_id = 0
        names = self.readNames()
        pbar = tqdm(desc="Processing", total=self.num_images)
        for idx in range(self.num_images):
            self.setImageIndex(idx)
            image = {}
            image['id'] = image_id
            image_id += 1
            image['width'] = self.viewer.camera.width
            image['height'] = self.viewer.camera.height
            image['file_name'] = str(idx) + self.image_format
            images.append(image)

            for i in range(self.listWidget.count()):
                ann = {}
                ann['id'] = ann_id
                ann_id += 1
                item = self.listWidget.item(i)
                self.listWidget.setCurrentItem(item)
                name = str(item.text())
                pose = list(self.viewer.get_pose_vec()) # x,y,z,qw,qx,qy,qz
                coords = self.viewer.bounding_box()
                width = coords[1] - coords[0]
                height = coords[3] - coords[2]
                area = width*height
                bbox = [coords[0], coords[2], width, height]
                ann['image_id'] = image['id']
                ann['category_id'] = names.index(name) 
                ann['iscrowd'] = 0
                ann['area'] = area
                ann['bbox'] = bbox
                ann['pose'] = pose
                ann['segmentation'] = []
                annotations.append(ann)
            pbar.update(1)
        pbar.close()
        ann_dict['images'] = images
        categories = [{"id": i, "name": name} for i, name in enumerate(names)]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(osp.join(self.image_folder,'..', json_name), 'wb') as outfile:
            outfile.write(json.dumps(ann_dict))

    def listSelectSlot(self, currItem, prevItem):
        if not currItem: return        
        objName = currItem.text()
        self.viewer.set_active(objName)
        alpha = self.viewer.get_alpha(objName)
        self.horizontalSlider_3.setSliderPosition(int(alpha*100))
        self.refreshAll()

    def alphaSliderMovedSlot(self, value):
        currItem = self.listWidget.currentItem()
        if not currItem: return
        objName = currItem.text()
        alpha = float(value)/100.0
        self.viewer.set_alpha(objName, alpha)
        self.refreshAll()


def parse_args():
    parser = argparse.ArgumentParser(description='Tool for annotating monocular image sequences with groundtruth poses.')
    parser.add_argument('--camera', type=str, required=True,
        help='Camera calibration yaml file in camchain format. Expected camera name is cam0.')
    parser.add_argument('--images', type=str, required=True,
        help='Folder containing image sequence. Expected naming convention is consecutive number sequence starting from 0.png.')
    parser.add_argument('--poses', type=str, required=True,
        help='CSV file containing pose annotations for image sequence. Expected format for each line in file is "image_number, x, y, z, qw, qx, qy, qz".')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MainWindowUIClass(args.camera, args.images, args.poses, MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())