"""
Reference URL: https://github.com/openvinotoolkit/cvat/blob/df175a856179f1c31ac2b15c80d98986a3d35acf/serverless/common/openvino/model_loader.py
"""
from openvino.inference_engine import IECore
import numpy as np

xTest = np.load("Data/xTest.npy")


def iou(box1, box2):
    intX0 = max(box1[0], box2[0])
    intY0 = max(box1[1], box2[1])
    intX1 = min(box1[2], box2[2])
    intY1 = min(box1[3], box2[3])

    intArea = max(intX1 - intX0, 0) * max(intY1 - intY0, 0)

    b1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intArea / (b1Area + b2Area - intArea + 1e-010)
    return iou


def getBoundingBoxes(result, confThresh=0.5, iouThresh=0.4):
    """
    :param result: 3D numpy array, first 4 values in 3rd dimension are bbox attrs, 5th is confidence
    :param confThresh: the threshold for deciding if prediction is valid
    :param iouThresh: the threshold for deciding if two boxes overlap
    """
    confMask = np.expand_dims(
        result[:, :, 4] > confThresh,
        axis=-1
    )
    result = result * confMask
    boundingBoxes = {}
    for pred in result:
        shape = pred.shape
        nonZeroIdxs = np.nonzero(pred)
        pred = pred[nonZeroIdxs]
        pred = pred.reshape(-1, shape[-1])
        boxes = pred[:, :5]
        classes = pred[:, 5:]
        classes = np.argmax(classes, axis=-1)
        unqClasses = set(classes)
        for unqClass in unqClasses:
            clsMask = classes == unqClass
            clsBoxes = boxes[np.nonzero(clsMask)]
            clsBoxes = clsBoxes[clsBoxes[:, -1].argsort()[::-1]]
            clsScores = clsBoxes[:, -1]
            clsBoxes = clsBoxes[:, :-1]
            boundingBoxes[unqClass] = boundingBoxes.get(unqClass, [])
            while len(clsBoxes) > 0:
                clsScore = clsScores[0]
                clsBox = clsBoxes[0]
                boundingBoxes[unqClass].append((clsBox, clsScore))
                clsBoxes = clsBoxes[1:]
                clsScores = clsScores[1:]
                ious = np.array([iou(clsBox, box) for box in clsBoxes], dtype=np.float32)
                iouMask = ious < iouThresh
                clsBoxes = clsBoxes[np.nonzero(iouMask)]
                clsScores = clsScores[np.nonzero(iouMask)]
    return boundingBoxes


core = IECore()
coreNet = core.read_network(
    model="saved_model.xml",
    weights="saved_model.bin"
)
net = core.load_network(coreNet, "CPU", num_requests=2)
inpBlobName = next(iter(coreNet.input_info))
outBlobName = next(iter(coreNet.outputs))
count = 0
for idx, img in enumerate(xTest):
    inputs = {inpBlobName: img.astype(np.float32)}
    results = net.infer(inputs)
    boundingBoxes = getBoundingBoxes(results[outBlobName])
    print(boundingBoxes)
