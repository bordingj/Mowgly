
from copy import deepcopy

from mowgly.training import loggers

from mowgly.training.generators import GeneratorFactory
from mowgly.model import Model


def early_stopping_train(model, generator_factory,
                         starting_patience, max_epochs, report_val_loss=True,
                         patience_increase_threshold=0.995, patience_multiplier=2.0,
                         logger=None):
    """
    This function performs early stopping training on a model.

    :type arrays: dict
    :param model: model to be trained.
    :type model: mowgly.model.Model
    :param generator_factory: generator factory that should make training and testing generators at each epoch
    :type generator_factory: mowgly.training.generators.GeneratorFactory
    :param loss_function: callable passed on to model.loss
    :type loss_function: callable
    :param score_function: callable passed on to model.score.
    :type score_function: callable.
    :param starting_patience: minimum number of epochs.
    :type starting_patience: int
    :param max_epochs: maximum number of epochs.
    :type max_epochs: int
    :param report_val_loss: whether to compute and report validation loss or not.
    :type report_val_loss: bool
    :param patience_increase_threshold: threshold for patience increase.
    :type patience_increase_threshold: float
    :param patience_multiplier: patience multiplier when patience is to be increased.
    :type patience_multiplier: float
    :param logger: a logger class that has method log_value (default: mowgly.training.loggers.ConsoleLogger)
    :param random_state: random number generator / seed (default: None).
    :return: best model found during training.
    """

    assert isinstance(generator_factory, GeneratorFactory)
    assert isinstance(model, Model)

    if logger is None:
        logger = loggers.ConsoleLogger()
    else:
        assert hasattr(logger, 'log_value')

    # get a validation generator and score model
    val_generator = generator_factory.make_val_generator()
    model.set_to_inference_mode()
    best_score = model.score(val_generator)
    logger.log_value('best_val_score', best_score, 0)

    best_model = deepcopy(model)
    scores_at_latest_patience_increase = best_score
    patience = starting_patience

    num_epochs = 0
    finished = False

    for epoch_number in range(max_epochs):

        # get a train generator and train the model for one epoch
        train_generator = generator_factory.make_train_generator()
        model.set_to_training_mode()
        train_loss = model.loss(train_generator)
        logger.log_value('train_loss', train_loss, epoch_number + 1)

        # get a validation generator and compute validation score
        val_generator = generator_factory.make_val_generator()
        model.set_to_inference_mode()
        current_score = model.score(val_generator)
        logger.log_value('current_val_score', current_score, epoch_number + 1)
        if report_val_loss:
            val_loss = model.loss(val_generator)
            logger.log_value('val_loss', val_loss, epoch_number + 1)

        if current_score > best_score:
            if patience_increase_threshold * current_score > scores_at_latest_patience_increase:
                patience = max( patience, patience_multiplier * num_epochs)
                scores_at_latest_patience_increase = current_score
            best_score = current_score
            best_model = deepcopy(model)

        logger.log_value('best_val_score', best_score, epoch_number + 1)

        if num_epochs >= patience:
            break

    return best_model
