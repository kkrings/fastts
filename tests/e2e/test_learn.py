from fastai.text.all import Learner  # type: ignore


def test_learn(learn: Learner) -> None:
    learn.fine_tune(1)
