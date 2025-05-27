from PIL import Image

ORIGINAL_WIDTH = 8192
ORIGINAL_HEIGHT = 4832
H_REPEATS = 4 # 宽度方向拼接4次
V_REPEATS = 7 # 高度方向拼接7次

original_image = Image.open('lunchuan.jpg')

final_width = ORIGINAL_WIDTH * H_REPEATS
final_height = ORIGINAL_HEIGHT * V_REPEATS

stitched_image = Image.new('RGB', (final_width, final_height))

for row in range(V_REPEATS):
    for col in range(H_REPEATS):
        x_offset = col * ORIGINAL_WIDTH
        y_offset = row * ORIGINAL_HEIGHT
        stitched_image.paste(original_image, (x_offset, y_offset))

stitched_image.save("jointlunchuan.jpg")

print(f"最终图像尺寸：{final_width}x{final_height}，已保存为 jointlunchuan.jpg")
print(f"最终宽度 {final_width} >= 30000 ({final_width >= 30000})")
print(f"最终高度 {final_height} >= 30000 ({final_height >= 30000})")