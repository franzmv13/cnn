import os
import tkinter as tk
from tkinter import Scrollbar
from PIL import Image, ImageTk
from main import prediction

model = "D:\school\comscie\emotion.V2\emotion_model20 ok.h5"
class ImageGalleryApp:
    def __init__(self, root, image_folder):
        self.root = root
        self.root.title("Image Gallery")

        self.image_folder = image_folder
        self.image_files = self.get_image_files()

        self.current_index = 0

        self.images_per_row = 12  # Number of images per row
        self.rows = (len(self.image_files) + self.images_per_row - 1) // self.images_per_row

        self.canvas = tk.Canvas(root)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = Scrollbar(root, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        self.frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.frame, anchor=tk.NW)

        self.show_images_in_grid()

    def get_image_files(self):
        image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        return image_files

    def show_images_in_grid(self):
        for i in range(len(self.image_files)):
            image_path = os.path.join(self.image_folder, self.image_files[i])
            image = Image.open(image_path)
            print(image_path)
            image.thumbnail((150, 150))
            tk_image = ImageTk.PhotoImage(image)

            image_label = tk.Label(self.frame, image=tk_image, text=prediction(image_path, model), compound=tk.BOTTOM)
            image_label.image = tk_image
            image_label.grid(row=i // self.images_per_row, column=i % self.images_per_row, padx=10, pady=10)

    def on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

if __name__ == "__main__":
    root = tk.Tk()
    image_folder_path = "D:\school\comscie\emotion.V2\\test_pics\happy"
    app = ImageGalleryApp(root, image_folder_path)
    root.mainloop()
