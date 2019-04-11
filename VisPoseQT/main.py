#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
if sys.platform == "linux" or sys.platform == "linux2":
    # TODO remove this OpenGL fix when PyQt
    # doesn't require OpenGL to be loaded first.
    # NOTE This must be placed before any other imports!
    import ctypes
    from ctypes.util import find_library
    libGL = find_library("GL")
    ctypes.CDLL(libGL, ctypes.RTLD_GLOBAL)

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtQml import *
from PyQt5.QtWidgets import *
import qml_qrc

def main():
    app = QApplication(sys.argv)

    engine = QQmlApplicationEngine()
    engine.load(QUrl("qrc:/main.qml"))

    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
