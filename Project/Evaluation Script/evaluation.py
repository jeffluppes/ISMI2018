# -*- coding: utf-8 -*-

from evalutils import DetectionEvaluation
from evalutils.io import CSVLoader
from evalutils.validators import ExpectedColumnNamesValidator
from evalutils.utils import score_detection


class Detection(DetectionEvaluation):
    def __init__(self):
        super().__init__(
            file_loader=CSVLoader(),
            validators=(
                ExpectedColumnNamesValidator(expected=(
                    "image_id_roi_id", "pixel_size", "x", "y", "score",
                )),
            ),
            join_key="image_id_roi_id",
        )
        self.detection_threshold = 0.0
        self.detection_radius = 4 / 0.24309392273426056

    def score_case(self, *, idx, case):
        ground_truth = case.loc['ground_truth']
        predictions = case.loc['predictions']

        ground_truth = [(p['x'], p['y']) for _, p in ground_truth.iterrows()
                        if p['score'] > self.detection_threshold]
        predictions = [(p['x'], p['y']) for _, p in predictions.iterrows()
                       if p['score'] > self.detection_threshold]

        if len(ground_truth) == 0:
            return {
                "false_positives": len(predictions),
                "false_negatives": 0,
                "true_positives": 0,
            }
        elif len(predictions) == 0:
            return {
                "false_positives": 0,
                "false_negatives": len(ground_truth),
                "true_positives": 0,
            }
        else:
            return score_detection(
                ground_truth=ground_truth,
                predictions=predictions,
                radius=self.detection_radius,
            )._asdict()


if __name__ == "__main__":
    evaluation = Detection()
    evaluation.evaluate()
