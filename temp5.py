import os

import cv2
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()

root.withdraw()

f_path = filedialog.askopenfilename()

image = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)

save_path_dir = os.path.join(os.path.dirname(f_path), 'output')
os.makedirs(save_path_dir, exist_ok=True)
save_path = os.path.join(save_path_dir, os.path.basename(f_path))

cv2.imwrite(save_path, image)
