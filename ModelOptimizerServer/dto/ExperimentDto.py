import pymysql

from dto.ModelDto import get_model_by_id
from dto.TestDto import TestDto
from utils import DB


class ExperimentDto:
    def __init__(self, exp_id, based_on, modification_text, model, state, date, sent_requests, tests_done, tests=None):
        self.exp_id = exp_id
        self.based_on = based_on
        self.modification_text = modification_text
        self.model = model  # ModelDto object
        self.state = state
        self.date = date
        self.sent_requests = sent_requests
        self.tests_done = tests_done
        self.tests = tests if tests else []  # List of TestDto objects

    def to_dict(self):
        return {
            "exp_id": self.exp_id,
            "based_on": self.based_on,
            "modification_text": self.modification_text,
            "model": self.model.to_dict() if self.model else None,
            "state": self.state,
            "date": self.date,
            "sent_requests": self.sent_requests,
            "tests_done": self.tests_done,
            "tests": [test.to_dict() for test in self.tests],
        }

    def __str__(self):
        tests_str = "\n".join([str(test) for test in self.tests])
        return f"Experiment ID: {self.exp_id}, State: {self.state}, Model: {self.model}, Tests:\n{tests_str}"


def get_experiment_by_id_with_tests(exp_id):
    """
    Fetches an experiment with its associated tests from the database.
    :param exp_id: The ID of the experiment to fetch.
    :return: ExperimentDto object if the experiment exists, otherwise None.
    """
    connection = None
    try:
        connection = DB.get_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        # Fetch experiment details
        cursor.execute("SELECT * FROM experiment WHERE exp_id = %s", (exp_id,))
        experiment_data = cursor.fetchone()

        if not experiment_data:
            return None  # Experiment not found

        # Fetch associated tests
        cursor.execute("SELECT * FROM test WHERE exp_id = %s", (exp_id,))
        tests_data = cursor.fetchall()

        # Construct TestDto objects
        tests = [
            TestDto(
                test_id=test["test_id"],
                test_name=test["test_name"],
                score=test["score"],
                date=test["date"],
                duration=test["duration"],
                hardware_config=test["hardware_config"],
                output_loss_epoch=test["output_loss_epoch"],
                output_accuracy_epoch=test["output_accuracy_epoch"]
            )
            for test in tests_data
        ]

        # Fetch associated model
        model = get_model_by_id(experiment_data["model_id"])
        if not model:
            return None  # Model not found

        # Construct and return ExperimentDto
        return ExperimentDto(
            exp_id=experiment_data["exp_id"],
            based_on=experiment_data["based_on"],
            modification_text=experiment_data["modification_text"],
            model=model,
            state=experiment_data["state"],
            date=experiment_data["date"],
            sent_requests=experiment_data["sent_requests"],
            tests_done=experiment_data["tests_done"],
            tests=tests
        )

    except Exception as e:
        print(f"Error fetching experiment: {e}")
        raise
    finally:
        if connection:
            connection.close()