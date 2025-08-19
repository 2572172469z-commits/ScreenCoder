import sys
import os
from PyQt5.QtWidgets import QApplication, QFileDialog, QPushButton, QVBoxLayout, QWidget, QMessageBox
import subprocess


class ScreenCoderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        btn_run = QPushButton('选择截图并生成HTML', self)
        btn_run.clicked.connect(self.run_workflow)
        layout.addWidget(btn_run)

        self.setLayout(layout)
        self.setWindowTitle('ScreenCoder GUI')

    def run_workflow(self):
        img_path, _ = QFileDialog.getOpenFileName(self, '选择截图', '', 'Images (*.png *.jpg)')
        if img_path:
            try:
                main_script = r"D:\app\ScreenCoder-main\main.py"
                output_dir = os.path.join(os.path.dirname(main_script), 'data', 'tmp')

                # 运行并检查返回码
                result = subprocess.run(
                    [sys.executable, main_script, '--screenshot', img_path],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    # 检查输出目录是否存在
                    if os.path.exists(output_dir):
                        # 获取生成的文件列表
                        generated_files = "\n".join(os.listdir(output_dir))

                        reply = QMessageBox.information(
                            self,
                            '处理完成',
                            f'处理成功！文件已生成在：\n{output_dir}\n\n'
                            f'生成的文件：\n{generated_files}\n\n'
                            '是否要打开输出目录？',
                            QMessageBox.Yes | QMessageBox.No
                        )

                        if reply == QMessageBox.Yes:
                            os.startfile(output_dir)  # Windows系统打开文件夹
                    else:
                        QMessageBox.warning(
                            self,
                            '警告',
                            f'处理完成但输出目录不存在：\n{output_dir}\n'
                            '可能是程序未按预期生成输出文件。'
                        )
                else:
                    error_msg = result.stderr if result.stderr else "未知错误，请查看控制台输出"
                    QMessageBox.critical(
                        self,
                        '错误',
                        f'生成HTML失败！\n错误信息：\n{error_msg}\n\n'
                        f'返回码：{result.returncode}'
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    '错误',
                    f'处理过程中出现错误:\n{str(e)}\n\n'
                    '请确保：\n'
                    '1. ScreenCoder主脚本路径正确\n'
                    '2. 所有依赖已安装\n'
                    '3. 输入图片格式正确'
                )


app = QApplication(sys.argv)
ex = ScreenCoderApp()
ex.show()
sys.exit(app.exec_())