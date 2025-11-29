[Pagina de los modelos de Yolo v3](https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html) (descargar **yolov3.weights**, **yolov3.cfg** y **coco.names**)

Crear la carpeta "model" y poner los archivos ahi

### Links directos

- [**yolov3.weights**](https://pjreddie.com/media/files/yolov3.weights)
- [**yolov3.cfg**](https://opencv-tutorial.readthedocs.io/en/latest/_downloads/10e685aad953495a95c17bfecd1649e5/yolov3.cfg)
- [**coco.names**](https://opencv-tutorial.readthedocs.io/en/latest/_downloads/a9fb13cbea0745f3d11da9017d1b8467/coco.names)

### Modelos Tiny para raspberry

**⚠ Importante:** Al usar estos modelos en la raspberry de manera local este puede llegar a temperaturas altas de mas de 60 grados, recomiendo un sistema de enfriamiento mejor

    wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
    wget https://pjreddie.com/media/files/yolov3-tiny.weights
