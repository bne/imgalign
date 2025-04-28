# imgalign

Dockerised homographic image alignernator, using OpenCV

```
docker run --rm -it \
    -v ./input:/app/input/ \
    -v ./output:/app/output/ \
    $(docker build -q .)
```

Image align python based on https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
