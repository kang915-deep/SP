"""Predeclared validation plateau; never treats an epoch cap as convergence."""


class Plateau:
    rates = (2e-4, 1e-4, 5e-5, 2.5e-5)

    def __init__(self):
        self.stage = 0
        self.last_progress = 0
        self.stage_epoch = 0
        self.loss = float("inf")
        self.gain = -float("inf")

    def update(self, epoch, loss, gain):
        loss_improved = loss < self.loss * (1 - 0.005)
        gain_improved = gain > self.gain + 0.1
        improved = loss_improved or gain_improved
        if improved:
            self.last_progress = epoch
        # Keep significance anchors: several small improvements must accumulate.
        # Updating these anchors every epoch would falsely call a slowly improving
        # curve a plateau merely because each individual increment is small.
        if loss_improved:
            self.loss = loss
        if gain_improved:
            self.gain = gain
        plateau = epoch >= 10 and epoch - max(self.last_progress, self.stage_epoch) >= 5
        action = "continue"
        if plateau:
            if self.stage == len(self.rates) - 1:
                action = "sufficient_validation_plateau"
            else:
                self.stage += 1
                self.stage_epoch = epoch
                self.last_progress = epoch
                self.loss = loss
                self.gain = gain
                action = "reduce_learning_rate"
        return {
            "action": action,
            "learning_rate": self.rates[self.stage],
            "stage": self.stage,
            "last_progress_epoch": self.last_progress,
            "stage_start_epoch": self.stage_epoch,
            "best_stage_loss": self.loss,
            "best_stage_gain": self.gain,
        }
