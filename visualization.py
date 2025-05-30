from keras_cv import visualization


def visualize_detections(model, dataset, bounding_box_format, cat_mapping):
    images, y_true = next(iter(dataset.take(5)))
    y_pred = model.predict(images)
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=2,
        show=True,
        font_scale=0.7,
        class_mapping=cat_mapping,
    )
