from yolo.dataset.annotation import parse_annotation


def main():
    annName = "/hdd/Datasets/counters/annotations/000020.xml"
    imgDir = "/hdd/Datasets/counters/img"
    labels = ["counter", "counter_screen"]
    imgFile, boxes, coded_labels = parse_annotation(annName, imgDir, labels)
    print(imgFile, boxes, coded_labels)


main()
