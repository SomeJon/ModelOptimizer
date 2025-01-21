import pymysql

from dto.ModelDto import get_model_by_id
from dto.TestDto import TestDto
from utils import DB


class TestJobDto:
    def __init__(self, model, test):
        self.model = model  # ModelDto object
        self.test = test  # TestDto object

    def to_dict(self):
        return {
            "model": self.model.to_dict(),
            "test": self.test.to_dict(),
        }

    def __str__(self):
        return f"Model:\n{self.model}\nTest:\n{self.test}"

def get_test_job_by_ids(model_id, test_id):
    """
    Fetches a TestJobDto from the database by model_id and test_id.
    :param model_id: The ID of the model to fetch.
    :param test_id: The ID of the test to fetch.
    :return: TestJobDto object if both model and test exist, otherwise None.
    """
    connection = None
    try:
        connection = DB.get_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        # Fetch the test details
        cursor.execute("SELECT * FROM test WHERE test_id = %s", (test_id,))
        test_data = cursor.fetchone()
        if not test_data:
            return None  # Test not found

        # Construct TestDto
        test = TestDto(
            test_id=test_data["test_id"],
            test_name=test_data["test_name"],
            score=test_data["score"],
            date=test_data["date"],
            duration=test_data["duration"],
            hardware_config=test_data["hardware_config"],
            output_loss_epoch=test_data["output_loss_epoch"],
            output_accuracy_epoch=test_data["output_accuracy_epoch"]
        )

        # Fetch the model using get_model_by_id
        model = get_model_by_id(model_id)
        if not model:
            return None  # Model not found

        # Combine into TestJobDto
        return TestJobDto(model=model, test=test)

    except Exception as e:
        print(f"Error fetching test job: {e}")
        raise
    finally:
        if connection:
            connection.close()