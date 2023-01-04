from easydict import EasyDict as edict

config = edict({
    "appIcon": "assets/app_icon.png",
    "imageDir": "../Annotation/images_annotation_1",
    "ImageDirLabelSize": 24,
    "ImageDirLabelWidth": 1920,
    "ImageDirLabelHeight": 50,


    "displayImageWidth": 800,
    "displayImageHeight":800,
    "ImageLabelSize:": 24,
})

if __name__ == "__main__":
    print(config)
