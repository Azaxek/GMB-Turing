from PIL import Image
import os

# Input/output directories
input_dir = "figures"
output_dir = "figures_tiff"
os.makedirs(output_dir, exist_ok=True)

# DPI and max width
DPI = 300
MAX_WIDTH_MM = 180
MAX_WIDTH_INCHES = MAX_WIDTH_MM / 25.4
MAX_WIDTH_PX = int(MAX_WIDTH_INCHES * DPI)

# Files to convert
files = [
    ("figure1.png", "Fig1.tiff"),
    ("figure2.png", "Fig2.tiff"),
    ("figure3.png", "Fig3.tiff"),
    ("figure4.png", "Fig4.tiff"),
    ("extended_data_fig1.png", "ExtendedData_Fig1.tiff"),
    ("extended_data_fig2.png", "ExtendedData_Fig2.tiff"),
    ("extended_data_fig3.png", "ExtendedData_Fig3.tiff"),
    ("graphical_abstract.png", "Graphical_Abstract.tiff"),
]

for src, dst in files:
    src_path = os.path.join(input_dir, src)
    dst_path = os.path.join(output_dir, dst)
    
    if not os.path.exists(src_path):
        print(f"Skipping {src} (not found)")
        continue
    
    img = Image.open(src_path)
    
    # Resize if too wide
    w, h = img.size
    if w > MAX_WIDTH_PX:
        ratio = MAX_WIDTH_PX / w
        new_size = (MAX_WIDTH_PX, int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        print(f"Resized {src} from {w}x{h} to {new_size[0]}x{new_size[1]}")
    
    # Convert to RGB if necessary (TIFF doesn't like RGBA)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Save as TIFF with DPI
    img.save(dst_path, format='TIFF', dpi=(DPI, DPI), compression='tiff_lzw')
    print(f"Saved {dst_path}")

print("All figures converted to TIFF.")
