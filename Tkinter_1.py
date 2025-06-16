import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

class ImageEnhancerApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Enhancer")
        master.geometry("1280x800")

        # Load trained model with custom metrics
        self.model = tf.keras.models.load_model('autoencoder.keras', 
                                              custom_objects={
                                                  'PSNR': self.psnr_metric,
                                                  'SSIM': self.ssim_metric
                                              })
        
        # Initialize GUI components
        self.create_widgets()
        self.input_image = None
        self.output_image = None

    def psnr_metric(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        max_pixel = tf.constant(1.0, dtype=tf.float32)
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        return 20 * tf.math.log(max_pixel / tf.sqrt(mse)) / tf.math.log(tf.constant(10.0, dtype=tf.float32))

    def ssim_metric(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.image.ssim(y_true, y_pred, 
                           max_val=tf.constant(1.0, dtype=tf.float32),
                           filter_size=11,
                           filter_sigma=1.5,
                           k1=0.01,
                           k2=0.03)

    def create_widgets(self):
        # Control Panel
        control_frame = tk.Frame(self.master)
        control_frame.pack(pady=10)

        self.upload_btn = tk.Button(control_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(side=tk.LEFT, padx=5)

        self.process_btn = tk.Button(control_frame, text="Enhance Image", command=self.process_image)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        self.process_btn.config(state=tk.DISABLED)

        self.download_btn = tk.Button(control_frame, text="Download Result", command=self.download_image)
        self.download_btn.pack(side=tk.LEFT, padx=5)
        self.download_btn.config(state=tk.DISABLED)

        # Image Display
        self.display_frame = tk.Frame(self.master)
        self.display_frame.pack(pady=20)

        self.input_label = tk.Label(self.display_frame, text="Input Image")
        self.input_label.grid(row=0, column=0, padx=10)

        self.output_label = tk.Label(self.display_frame, text="Enhanced Image")
        self.output_label.grid(row=0, column=1, padx=10)

        self.input_canvas = tk.Canvas(self.display_frame, width=600, height=400)
        self.input_canvas.grid(row=1, column=0, padx=10)

        self.output_canvas = tk.Canvas(self.display_frame, width=600, height=400)
        self.output_canvas.grid(row=1, column=1, padx=10)

        # Metrics Display
        self.metrics_frame = tk.Frame(self.master)
        self.metrics_frame.pack(pady=10)

        self.psnr_label = tk.Label(self.metrics_frame, text="PSNR: --")
        self.psnr_label.pack(side=tk.LEFT, padx=20)

        self.ssim_label = tk.Label(self.metrics_frame, text="SSIM: --")
        self.ssim_label.pack(side=tk.LEFT, padx=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            # Load and resize to model's expected input shape (800x1200)
            self.input_image = Image.open(file_path).resize((1200, 800))  # PIL uses (width, height)
            self.show_image(self.input_image, self.input_canvas)
            self.process_btn.config(state=tk.NORMAL)

    def process_image(self):
        if self.input_image:
            # Convert to float32 array and normalize
            img_array = np.array(self.input_image, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Model prediction (output in [0,1] range)
            output_array = self.model.predict(img_array, verbose=0)[0]
            
            # Convert to uint8 for image display
            output_uint8 = np.clip(output_array * 255, 0, 255).astype(np.uint8)
            self.output_image = Image.fromarray(output_uint8)
            self.show_image(self.output_image, self.output_canvas)
            self.download_btn.config(state=tk.NORMAL)

            # Calculate metrics using original float32 arrays
            y_true = img_array[0]  # Already in [0,1] range
            y_pred = output_array  # Direct model output in [0,1]
            
            psnr = self.psnr_metric(y_true, y_pred).numpy()
            ssim = self.ssim_metric(y_true, y_pred).numpy()
            
            self.psnr_label.config(text=f"PSNR: {psnr:.2f} dB")
            self.ssim_label.config(text=f"SSIM: {ssim:.4f}")

    def show_image(self, image, canvas):
        """Display image on specified canvas"""
        canvas.delete("all")
        # Resize for display while maintaining aspect ratio
        display_img = image.resize((600, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display_img)
        canvas.image = photo  # Keep reference to prevent garbage collection
        canvas.create_image(300, 200, image=photo)

    def download_image(self):
        if self.output_image:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
            )
            if save_path:
                # Save full resolution output
                full_res_output = Image.fromarray(np.clip(self.model.predict(
                    np.expand_dims(np.array(self.input_image, dtype=np.float32)/255.0, axis=0)
                )[0] * 255, 0, 255).astype(np.uint8))
                
                full_res_output.save(save_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEnhancerApp(root)
    root.mainloop()
