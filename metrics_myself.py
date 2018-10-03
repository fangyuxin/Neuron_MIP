import numpy as np

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def reset(self):
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred):
        hist = np.bincount(
            self.n_classes * label_true + label_pred,
            minlength=self.n_classes).reshape(self.n_classes, self.n_classes)
        return hist

    def update(self, labels_true, labels_pred):
        for label_true, label_pred in zip(labels_true, labels_pred):
            self.confusion_matrix += self._fast_hist(
                label_true.flatten(), label_pred.flatten())

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))


        score = {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                }

        return score



    '''Test'''


    running_score = runningScore(2)

    label_true = np.array([[0,0,1],
                          [1,1,0],
                          [0,1,0]])

    label_pred = np.array([[0,0,1],
                           [1,1,0],
                           [0,1,0]])

    running_score.update(label_true, label_pred)

    running_score.get_scores()
