class TestDto:
    def __init__(self, test_id, test_name, score, date, duration, hardware_config, output_loss_epoch, output_accuracy_epoch):
        self.test_id = test_id
        self.test_name = test_name
        self.score = score
        self.date = date
        self.duration = duration
        self.hardware_config = hardware_config
        self.output_loss_epoch = output_loss_epoch
        self.output_accuracy_epoch = output_accuracy_epoch

    def to_dict(self):
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "score": self.score,
            "date": self.date,
            "duration": self.duration,
            "hardware_config": self.hardware_config,
            "output_loss_epoch": self.output_loss_epoch,
            "output_accuracy_epoch": self.output_accuracy_epoch,
        }

    def __str__(self):
        return (
            f"Test ID: {self.test_id}, Name: {self.test_name}, Score: {self.score}, Date: {self.date}, "
            f"Duration: {self.duration}, Hardware Config: {self.hardware_config}"
        )
