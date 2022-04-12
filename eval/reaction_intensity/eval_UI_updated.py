import random
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
        self.EditOpenPath = QLineEdit()  # edit line to show the path of opening image
        self.EditOpenPath.setEnabled(False)
        self.NextImgButton = QPushButton("Next Image")  # switch to the next image
        self.DisagreeSegButton = QPushButton("Disagree Segmentation")  # disagree segmentation button
        self.EditSavePath = QLineEdit()  # edit line to show the path of saving results
        self.LabelSavePath = QLabel("Save to Directory: ")
        self.EditSavePath.setEnabled(False)
        self.EditAddNotes = QLineEdit()
        self.LabelAddNotes = QLabel("Notes: ")

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
        self.EditAddNotes.setPlaceholderText("Add notes here...")

        self.AllImgfnList = []
        self.seg = []
        self.fib_score = []
        self.cel_score = []
        self.ori_score = []
        self.notes = []
        self.img_idx = 0
        pixmap_f = QPixmap(1124, 768)
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
        row4_hbox = QHBoxLayout()

        row1_hbox.addWidget(self.OpenCaseButton)
        row1_hbox.addWidget(self.EditOpenPath)

        row2_vbox = QVBoxLayout()
        seg_row = QVBoxLayout()
        fib_row = QVBoxLayout()
        cel_row = QVBoxLayout()
        ori_row = QVBoxLayout()
        seg_row.addWidget(self.DisagreeSegButton)
        fib_row.addWidget(self.fibCheckBox1)
        fib_row.addWidget(self.fibCheckBox2)
        fib_row.addWidget(self.fibCheckBox3)
        cel_row.addWidget(self.celCheckBox1)
        cel_row.addWidget(self.celCheckBox2)
        cel_row.addWidget(self.celCheckBox3)
        ori_row.addWidget(self.oriCheckBox1)
        ori_row.addWidget(self.oriCheckBox2)
        ori_row.addWidget(self.oriCheckBox3)
        row2_vbox.addLayout(seg_row)
        row2_vbox.addLayout(fib_row)
        row2_vbox.addLayout(cel_row)
        row2_vbox.addLayout(ori_row)

        row2_hbox.addWidget(self.ResultImageLabel)
        row2_hbox.addLayout(row2_vbox)

        row3_hbox.addWidget(self.LabelSavePath)
        row3_hbox.addWidget(self.EditSavePath)
        row3_hbox.addWidget(self.NextImgButton)

        row4_hbox.addWidget(self.LabelAddNotes)
        row4_hbox.addWidget(self.EditAddNotes)

        vbox_main.addLayout(row1_hbox)
        vbox_main.addLayout(row2_hbox)
        vbox_main.addLayout(row3_hbox)
        vbox_main.addLayout(row4_hbox)

        vbox_main.setSizeConstraint(QLayout.SetMinimumSize)
        self.setLayout(vbox_main)

    def load(self, idx):
        ImgRes = Image.open(self.AllImgfnList[idx])
        self.EditOpenPath.setText(self.AllImgfnList[idx])
        pixmap_fix = ImageQt(ImgRes)
        self.ResultPixmap = QPixmap.fromImage(pixmap_fix)
        self.ResultImageLabel.setPixmap(self.ResultPixmap)

        self.seg.append(True)
        self.fib_score.append(-1)  # TODO:
        self.cel_score.append(-1)  # TODO:
        self.ori_score.append(-1)  # TODO:
        self.notes.append(self.EditAddNotes.text())
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
        if len(self.AllImgfnList) > self.img_idx + 1:
            self.img_idx += 1
            self.load(self.img_idx)
            self.EditAddNotes.setText("")
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
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            input_dir = str(folder)
            parent_path = os.path.split(input_dir)[0]
            self.EditSavePath.setText(parent_path)
            self.AllImgfnList = glob.glob(os.path.join(input_dir, "*.png"))
            total_cnt = len(self.AllImgfnList)
            path = os.path.split(self.AllImgfnList[0])[0]
            r_dir = os.path.split(path)[0]
            csv_fn = os.path.join(r_dir, os.path.split(path)[1] + "_eval.csv")
            # self.SelectedImgfnList = self.AllImgfnList
            cnt = 0
            if os.path.exists(csv_fn):
                fp = open(csv_fn, 'r')
                lines = fp.readlines()
                for l in lines:
                    ele = l.split(",")
                    if ele[0] in self.AllImgfnList:
                        self.AllImgfnList.remove(ele[0])
                        cnt += 1
                fp.close()
            #random.shuffle(self.AllImgfnList)
            if len(self.AllImgfnList) > 0:
                self.img_idx = 0
                self.load(0)
            QMessageBox.information(self, 'Info', '%d images in folder, %d previously evaluated image(s) have been excluded' % (total_cnt, cnt),
                                         QMessageBox.Ok)
        else:
            QMessageBox.question(self, 'Info',
                                 'No folder is selected, please select again.',
                                 QMessageBox.Ok)


    def save_info(self):
        if len(self.seg) > 0:
            path = os.path.split(self.AllImgfnList[0])[0]
            r_dir = os.path.split(path)[0]
            csv_fn = os.path.join(r_dir, os.path.split(path)[1] + "_eval.csv")
            if os.path.exists(csv_fn):
                fp = open(csv_fn, 'r')
                str_wrt = fp.read()
                fp.close()
                fp = open(csv_fn, 'a')
            else:
                fp = open(csv_fn, 'w')
                str_wrt = "Img_fn,Segmentation,Fibrosis,Cellularity,Orientation,Notes\n"
            for idx, s in enumerate(self.seg):
                str_wrt += self.AllImgfnList[idx] + "," + str(s) + "," + str(self.fib_score[idx]) \
                           + "," + str(self.cel_score[idx]) + "," + str(self.ori_score[idx]) + "," + self.notes[idx] + "\n"
            
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
