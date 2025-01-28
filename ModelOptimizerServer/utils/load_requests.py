import json
from datetime import datetime
from utils.DB import DB


def load_results(data):
    connection = DB.get_connection()
    cursor = connection.cursor()

    try:
        for result in data:
            # Determine the training date
            training_date_str = result.get('train_stats', {}).get('training_date')
            if training_date_str:
                try:
                    training_date = datetime.strptime(training_date_str, '%Y-%m-%dT%H:%M:%S.%f')
                except ValueError:
                    training_date = datetime.now()
            else:
                # Use execution_timestamp if training_date is not available
                exec_ts_str = result.get('execution_timestamp')
                if exec_ts_str:
                    try:
                        training_date = datetime.strptime(exec_ts_str, '%Y-%m-%dT%H:%M:%S.%f')
                    except ValueError:
                        training_date = datetime.now()
                else:
                    training_date = datetime.now()
            formatted_training_date = training_date.strftime('%Y-%m-%d %H:%M:%S')

            print(formatted_training_date)

            # Extract fields with appropriate default values
            test_id = result.get('test_id')
            exp_id = result.get('exp_id')
            score = result.get('test_stats', {}).get('accuracy')
            duration_seconds = result.get('train_stats', {}).get('training_time_seconds')
            hardware_config = result.get('device_name')
            output_loss_epoch = json.dumps(result.get('train_stats', {}).get('epoch_losses', []))
            output_accuracy_epoch = json.dumps(result.get('train_stats', {}).get('epoch_accuracies', []))
            mse = result.get('test_stats', {}).get('mse')
            variance_dataset = result.get('test_stats', {}).get('variance_dataset')
            variance_y_hat = result.get('test_stats', {}).get('variance_y_hat')
            mean_bias = result.get('test_stats', {}).get('mean_bias')
            model_architecture = result.get('model_architecture')
            error_message = result.get('error_message')

            # Prepare the data tuple, ensuring all fields are correctly set
            data_tuple = (
                test_id,
                exp_id,
                score,  # This should be a float or None
                formatted_training_date,
                duration_seconds,  # Float or None
                hardware_config,  # String or None
                output_loss_epoch,  # JSON string
                output_accuracy_epoch,  # JSON string
                mse,  # Float or None
                variance_dataset,  # Float or None
                variance_y_hat,  # Float or None
                mean_bias,  # Float or None
                model_architecture,  # String or None
                error_message  # String or None
            )

            # Debugging: Print the data tuple
            print("Data Tuple:", data_tuple)

            # Update test record in the database
            query_test = """
                INSERT INTO test (test_id, exp_id, score, date, duration_seconds, hardware_config,
                                  output_loss_epoch, output_accuracy_epoch, mse, variance_dataset,
                                  variance_y_hat, mean_bias, model_architecture, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                score=VALUES(score), date=VALUES(date), duration_seconds=VALUES(duration_seconds),
                hardware_config=VALUES(hardware_config), output_loss_epoch=VALUES(output_loss_epoch),
                output_accuracy_epoch=VALUES(output_accuracy_epoch), mse=VALUES(mse),
                variance_dataset=VALUES(variance_dataset), variance_y_hat=VALUES(variance_y_hat),
                mean_bias=VALUES(mean_bias), model_architecture=VALUES(model_architecture),
                error_message=VALUES(error_message)
            """
            cursor.execute(query_test, data_tuple)

            # Determine the new state for the experiment
            status = result.get('status')
            if status == 'Success':
                new_state = 'Completed'
            elif status == 'Failed':
                new_state = 'Failed'
            else:
                new_state = 'Partial'  # Handle other potential states

            # Update experiment record in the database
            query_exp = """
                UPDATE experiment SET state=%s, tests_done=tests_done+1 WHERE exp_id=%s
            """
            cursor.execute(query_exp, (new_state, exp_id))

        connection.commit()

    except Exception as e:
        if connection:
            connection.rollback()
        print("Error in load_results:", e)
        raise

    finally:
        if connection:
            connection.close()
