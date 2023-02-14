import sys
# 从转换的.py文件内调用类
from My_Setting_UI import Ui_MainWindow
from PyQt5 import QtWidgets


class myWin(QtWidgets.QWidget, Ui_MainWindow):

    def __init__(self):
        super(myWin, self).__init__()
        self.setupUi(self)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Widget = myWin()
    Widget.show()
    sys.exit(app.exec_())