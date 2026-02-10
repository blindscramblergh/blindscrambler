import torch
import numpy as np
import matplotlib.pyplot as plt
from manim._config import config
from manim.constants import UP, DOWN, LEFT, RIGHT, RESAMPLING_ALGORITHMS


from manim.mobject.types.image_mobject import ImageMobject

# Colors (Imported directly from the color module)
from manim.utils.color import WHITE, BLUE, RED, interpolate_color

# Rate Functions (For animation smoothing)
from manim.utils.rate_functions import linear

# Mobjects
from manim.mobject.geometry.polygram import Square, Rectangle
from manim.mobject.text.text_mobject import Text
from manim.mobject.text.numbers import DecimalNumber
from manim.mobject.types.vectorized_mobject import VGroup
from manim.scene.scene import Scene

class LargeWeightMatrixAnime(Scene):
    def __init__(self, matrix_stack, dt=0.2, title_str="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Move to CPU/NumPy immediately for massive speed gains
        self.data = matrix_stack.detach().cpu().numpy()
        self.dt = dt
        self.title_str = title_str

    def construct(self):
        steps, rows, cols = self.data.shape
        v_min, v_max = self.data.min(), self.data.max()
        
        if self.title_str:
            header = Text(self.title_str, font_size=36).to_edge(UP, buff=0.5)
            self.add(header)

        # Helper to convert a matrix frame into a colormapped ImageMobject
        def get_image_mobject(matrix_data):
            # 1. Normalize to [0, 1] using global bounds
            norm = (matrix_data - v_min) / (v_max - v_min + 1e-8)
            # 2. Apply a colormap (RdBu_r goes from Blue to Red)
            # This returns an RGBA array
            color_indices = plt.cm.RdBu_r(norm)
            # 3. Convert to RGB uint8 (0-255) for Manim
            rgb_data = (color_indices[:, :, :3] * 255).astype(np.uint8)
            
            img = ImageMobject(rgb_data)
            img.height = 6  # Scale image to fit the screen vertically
            img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            img.shift(LEFT * 1.5)
            return img

        # Static Color Bar
        color_bar = Rectangle(height=5, width=0.5, stroke_color=WHITE, stroke_width=2)
        color_bar.set_fill(color=[BLUE, RED], opacity=1).to_edge(RIGHT, buff=1.5)
        
        max_label = DecimalNumber(v_max, num_decimal_places=1).scale(0.7).next_to(color_bar, UP)
        min_label = DecimalNumber(v_min, num_decimal_places=1).scale(0.7).next_to(color_bar, DOWN)
        self.add(color_bar, max_label, min_label)

        # Initial Image
        current_img = get_image_mobject(self.data[0])
        self.add(current_img)
        self.wait(self.dt)

        # Animation Loop
        for i in range(1, steps):
            new_img = get_image_mobject(self.data[i])
            # For ImageMobjects, 'become' is very efficient
            self.play(
                current_img.animate.become(new_img),
                run_time=self.dt,
                rate_func=linear
            )

def animate_large_heatmap(matrix_stack, dt=0.2, resolution=(1920, 1080), file_name="large_wt_anim", title_str=""):
    config.pixel_width, config.pixel_height = resolution
    config.frame_rate = 30
    config.disable_caching = True
    config.output_file = file_name

    scene = LargeWeightMatrixAnime(matrix_stack, dt=dt, title_str=title_str)
    scene.render()

if __name__ == "__main__":
    print("Generating matrix...")
    large_data = torch.randn(20, 1000, 1000) 
    
    print("Starting render...")
    animate_large_heatmap(
        large_data, 
        dt=0.04,
        file_name="large_weights_evolution",
        title_str="Weight Evolution"
    )