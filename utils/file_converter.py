from PIL import Image

path_to_png = r"C:\Users\erikc\Documents\Paper\Paper\CVPR2027\method overview 2.png"
with Image.open(path_to_png) as image:
    image = image.convert("RGBA")
    background = Image.new("RGB", image.size, "white")
    background.paste(image, mask=image.getchannel("A"))
    background.save("image.pdf", "PDF", resolution=600.0)
