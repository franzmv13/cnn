import tkinter as tk
from PIL import Image, ImageTk
from main import prediction

model = "D:\school\comscie\emotion.V2\emotion_model20 ok.h5"
def print_user_input():
    user_input = entry.get().replace('"',"")
    print(user_input)
    try:
        # Try to open the image
        img = Image.open(user_input)
        
        # Display the image on the right side
        img.thumbnail((500, 500))  # Resize the image for display
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img  # Keep a reference to the image to prevent it from being garbage collected

        # Update the result label
        result_label.config(text=f"Output: {prediction(user_input,model)}")
    except Exception as e:
        # If there's an error loading the image, show an error message
        result_label.config(text=f"Error: {e}")
        img_label.config(image=None)

# Create the main window
window = tk.Tk()
window.title("Emotion Prediction")

# Create an entry widget to accept user input
entry = tk.Entry(window, width=60)
entry.pack(pady=10)

# Create a button to trigger printing of user input
print_button = tk.Button(window, text="Test", command=print_user_input)
print_button.pack(pady=10)

# Create a label to display the result
result_label = tk.Label(window, text="")
result_label.pack(pady=10)

# Create a label to display the image
img_label = tk.Label(window)
img_label.pack(padx=20)

# Start the main event loop
window.mainloop()
