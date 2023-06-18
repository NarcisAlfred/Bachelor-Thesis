from cx_Freeze import setup, Executable
import sys
sys.setrecursionlimit(5000000)

setup(
    name="Echocardiography Segmentation",
    version="1.0",
    description="App used to predict segmentation of echocardiography images",
    executables=[Executable("MainWindow.pyw")],
)