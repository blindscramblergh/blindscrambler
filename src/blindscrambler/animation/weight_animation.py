import torch
from manim._config import config
from manim.constants import UP, DOWN, LEFT, RIGHT

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

class WeightMatrixAnime(Scene):
    
    def __init__(self, matrix_stack, dt=0.2, resolution=(1080, 1080), title_str="", *args, **kwargs):
        super(WeightMatrixAnime, self).__init__(*args, **kwargs)
    
        self.matrix_stack = matrix_stack
        self.dt = dt
        self.resolution =resolution
        self.title_str = title_str
    
    
    def construct(self):
        steps, rows, cols = self.matrix_stack.shape

        # 1. Global Bounds from the entire 3D stack
        v_min = float(self.matrix_stack.min())
        v_max = float(self.matrix_stack.max())
        denom = v_max - v_min + 1e-6

        # Optional Title Logic
        if self.title_str:
            header = Text(self.title_str, font_size=24).to_edge(UP, buff=-0.4)
            self.add(header)

        def create_grid(matrix_data):
            grid = VGroup()
            # Dynamically adjust cell size if matrix is large
            cell_size = 6.0 / max(rows, cols) 
            if cell_size > 1.0: cell_size = 1.0

            for r in range(rows):
                for c in range(cols):
                    val = float(matrix_data[r, c])
                    alpha = max(0, min(1, (val - v_min) / denom))
                    color_val = interpolate_color(BLUE, RED, alpha)

                    square = Square(
                        side_length=cell_size,
                        fill_color=color_val,
                        fill_opacity=1,
                        stroke_width=0.5
                    )
                    square.move_to([c * cell_size, -r * cell_size, 0])
                    grid.add(square)
            
            # Shift left to accommodate color bar on the right
            return grid.center().shift(LEFT * 1.5) # Shift left to make room for color bar

        # 2. Create the Color Bar (to the right)
        color_bar = Rectangle(
            height=5, width=0.5,
            stroke_color=WHITE, stroke_width=2
        )
        color_bar.set_fill(color=[BLUE, RED], opacity=1)
        color_bar.to_edge(RIGHT, buff=1.5)

        # 3. Add Labels to Color Bar
        max_label = DecimalNumber(v_max, num_decimal_places=1).scale(0.7)
        min_label = DecimalNumber(v_min, num_decimal_places=1).scale(0.7)
        max_label.next_to(color_bar, UP)
        min_label.next_to(color_bar, DOWN)

        bar_group = VGroup(color_bar, max_label, min_label)

        # Initial setup
        current_grid = create_grid(self.matrix_stack[0])
        self.add(current_grid, bar_group)
        self.wait(self.dt)

        # Animation loop
        for i in range(1, steps):
            new_grid = create_grid(self.matrix_stack[i])
            self.play(
                current_grid.animate.become(new_grid),
                run_time=self.dt,
                rate_func=linear
            )
        self.wait(1)

def generate_weight_history(steps=30, size=8):
    weights = torch.zeros(steps, size, size)
    current = torch.randn(size, size)
    for i in range(steps):
        current += torch.randn(size, size) * 5 + 10
        weights[i] = current.clone()
    return weights

def animate_weight_heatmap(matrix_stack, dt=0.2, resolution=(1080, 1080), file_name="weight_animation", title_str=""):
    config.pixel_width = resolution[0]
    config.pixel_height = resolution[1]
    config.frame_rate = 30
    config.disable_caching = True
    config.max_files_cached = 1000000
    
    # Specify the file name (do not include .mp4, Manim adds it)
    config.output_file = file_name

    # Create an object of WeightMatrixAnime
    scene = WeightMatrixAnime(matrix_stack, dt=dt,  title_str=title_str)
    scene.render()

if __name__ == "__main__":
    # Example with a 20x20 matrix and a title
    #weight_data = generate_weight_history(steps=30, size=20)
    
    weight_data = torch.randn(20, 50, 50)
    animate_weight_heatmap(
        weight_data, 
        dt=0.04,
        file_name = "wt_animation",
        title_str="Neural Weight Distribution"
    )