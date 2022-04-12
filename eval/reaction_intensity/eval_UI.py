import sys, os
import glob
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPixmap, QCursor, QPen
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QApplication,
                             QFileDialog, QLabel, QMessageBox, QLineEdit, QLayout, QCheckBox)


class EvalTSR(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.OpenCaseButton = QPushButton("Open a Folder")  # open images from a case
        self.NextImgButton = QPushButton("Next Image")  # switch to the next image
        self.DisagreeSegButton = QPushButton("Disagree Segmentation")  # disagree segmentation button
        self.EditSavePath = QLineEdit()  # edit line to show the path of saving results
        self.LabelSavePath = QLabel("Save to Directory: ")
        self.EditSavePath.setEnabled(False)

        self.ResultImageLabel = QLabel()  # show images
        self.fibCheckBox1 = QCheckBox("low", self)
        self.fibCheckBox2 = QCheckBox("medium", self)
        self.fibCheckBox3 = QCheckBox("high", self)
        self.celCheckBox1 = QCheckBox("low", self)
        self.celCheckBox2 = QCheckBox("medium", self)
        self.celCheckBox3 = QCheckBox("high", self)
        self.oriCheckBox1 = QCheckBox("low", self)
        self.oriCheckBox2 = QCheckBox("medium", self)
        self.oriCheckBox3 = QCheckBox("high", self)

        self.ImgfnList = []
        self.seg = []
        self.fib_score = []
        self.cel_score = []
        self.ori_score = []
        self.img_idx = 0
        pixmap_f = QPixmap(1280, 256)
        pixmap_f.fill(Qt.white)
        self.ResultPixmap = pixmap_f
        self.ResultImageLabel.setPixmap(self.ResultPixmap)

        self.OpenCaseButton.clicked.connect(self.OpenCaseDirDialog)
        self.NextImgButton.clicked.connect(self.next_img)
        self.DisagreeSegButton.clicked.connect(self.disagree)
        self.fibCheckBox1.stateChanged.connect(self.uncheck)
        self.fibCheckBox2.stateChanged.connect(self.uncheck)
        self.fibCheckBox3.stateChanged.connect(self.uncheck)
        self.celCheckBox1.stateChanged.connect(self.uncheck)
        self.celCheckBox2.stateChanged.connect(self.uncheck)
        self.celCheckBox3.stateChanged.connect(self.uncheck)
        self.oriCheckBox1.stateChanged.connect(self.uncheck)
        self.oriCheckBox2.stateChanged.connect(self.uncheck)
        self.oriCheckBox3.stateChanged.connect(self.uncheck)

        vbox_main = QVBoxLayout()
        row1_hbox = QHBoxLayout()
        row2_hbox = QHBoxLayout()
        row3_hbox = QHBoxLayout()

        fib_row = QVBoxLayout()
        cel_row = QVBoxLayout()
        ori_row = QVBoxLayout()

        row1_hbox.addWidget(self.OpenCaseButton)
        row1_hbox.addWidget(self.DisagreeSegButton)
        fib_row.addWidget(self.fibCheckBox1)
        fib_row.addWidget(self.fibCheckBox2)
        fib_row.addWidget(self.fibCheckBox3)
        cel_row.addWidget(self.celCheckBox1)
        cel_row.addWidget(self.celCheckBox2)
        cel_row.addWidget(self.celCheckBox3)
        ori_row.addWidget(self.oriCheckBox1)
        ori_row.addWidget(self.oriCheckBox2)
        ori_row.addWidget(self.oriCheckBox3)

        row1_hbox.addLayout(fib_row)
        row1_hbox.addLayout(cel_row)
        row1_hbox.addLayout(ori_row)

        row2_hbox.addWidget(self.ResultImageLabel)
        row2_hbox.addWidget(self.NextImgButton)

        row3_hbox.addWidget(self.LabelSavePath)
        row3_hbox.addWidget(self.EditSavePath)

        vbox_main.addLayout(row1_hbox)
        vbox_main.addLayout(row2_hbox)
        vbox_main.addLayout(row3_hbox)
        vbox_main.setSizeConstraint(QLayout.SetMinimumSize)
        self.setLayout(vbox_main)

    def load(self, idx):
        ImgRes = Image.open(self.ImgfnList[idx])
        pixmap_fix = ImageQt(ImgRes)
        self.ResultPixmap = QPixmap.fromImage(pixmap_fix)
        self.ResultImageLabel.setPixmap(self.ResultPixmap)

        self.seg.append(True)
        self.fib_score.append(-1)  # TODO:
        self.cel_score.append(-1)  # TODO:
        self.ori_score.append(-1)  # TODO:
        self.fibCheckBox1.setChecked(False)
        self.fibCheckBox2.setChecked(False)
        self.fibCheckBox3.setChecked(False)
        self.celCheckBox1.setChecked(False)
        self.celCheckBox2.setChecked(False)
        self.celCheckBox3.setChecked(False)
        self.oriCheckBox1.setChecked(False)
        self.oriCheckBox2.setChecked(False)
        self.oriCheckBox3.setChecked(False)


    def next_img(self):
        if len(self.ImgfnList) > self.img_idx + 1:
            self.img_idx += 1
            self.load(self.img_idx)
        else:
            QMessageBox.information(self, 'Message', "No more images in this case", QMessageBox.Ok)

    def disagree(self):
        value = self.seg[self.img_idx]
        self.seg[self.img_idx] = not value

    # uncheck method
    def uncheck(self, state):
        if state == Qt.Checked:
            if self.sender() == self.fibCheckBox1:
                self.fib_score[self.img_idx] = 0
                self.fibCheckBox2.setChecked(False)
                self.fibCheckBox3.setChecked(False)
            elif self.sender() == self.fibCheckBox2:
                self.fib_score[self.img_idx] = 1
                self.fibCheckBox1.setChecked(False)
                self.fibCheckBox3.setChecked(False)
            elif self.sender() == self.fibCheckBox3:
                self.fib_score[self.img_idx] = 2
                self.fibCheckBox1.setChecked(False)
                self.fibCheckBox2.setChecked(False)
            #####
            if self.sender() == self.celCheckBox1:
                self.cel_score[self.img_idx] = 0
                self.celCheckBox2.setChecked(False)
                self.celCheckBox3.setChecked(False)
            elif self.sender() == self.celCheckBox2:
                self.cel_score[self.img_idx] = 1
                self.celCheckBox1.setChecked(False)
                self.celCheckBox3.setChecked(False)
            elif self.sender() == self.celCheckBox3:
                self.cel_score[self.img_idx] = 2
                self.celCheckBox1.setChecked(False)
                self.celCheckBox2.setChecked(False)
            #####
            if self.sender() == self.oriCheckBox1:
                self.ori_score[self.img_idx] = 0
                self.oriCheckBox2.setChecked(False)
                self.oriCheckBox3.setChecked(False)
            elif self.sender() == self.oriCheckBox2:
                self.ori_score[self.img_idx] = 1
                self.oriCheckBox1.setChecked(False)
                self.oriCheckBox3.setChecked(False)
            elif self.sender() == self.oriCheckBox3:
                self.ori_score[self.img_idx] = 2
                self.oriCheckBox1.setChecked(False)
                self.oriCheckBox2.setChecked(False)

    def OpenCaseDirDialog(self):
        input_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        parent_path = os.path.split(input_dir)[0]
        self.EditSavePath.setText(parent_path)
        self.ImgfnList = glob.glob(os.path.join(input_dir, "*.png"))
        if len(self.ImgfnList) > 0:
            self.img_idx = 0
            self.load(0)

    def save_info(self):
        if len(self.seg) > 0:
            path = os.path.split(self.ImgfnList[0])[0]
            r_dir = os.path.split(path)[0]
            csv_fn = os.path.join(r_dir, os.path.split(path)[1] + "_eval.csv")
            str_wrt = "Img_fn,Segmentation,Fibrosis,Cellularity,Orientation\n"
            for idx, s in enumerate(self.seg):
                str_wrt += self.ImgfnList[idx] + "," + str(s) + "," + str(self.fib_score[idx]) \
                           + "," + str(self.cel_score[idx]) + "," + str(self.ori_score[idx]) + "\n"
            fp = open(csv_fn, 'w')
            fp.write(str_wrt)
            fp.close()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            self.save_info()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Main_Window = EvalTSR()
    Main_Window.show()
    sys.exit(app.exec_())
