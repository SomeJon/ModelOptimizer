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
            score = result.get('test_stats', {}).get('test_accuracy', 0)
            duration_seconds = result.get('train_stats', {}).get('training_time_seconds', 0)
            hardware_config = result.get('device_name')

            epoch_losses_train = json.dumps(result.get('train_stats', {}).get('epoch_losses_train', []))
            epoch_losses_validation = json.dumps(result.get('train_stats', {}).get('epoch_losses_validation', []))
            epoch_accuracies_train = json.dumps(result.get('train_stats', {}).get('epoch_accuracies_train', []))
            epoch_accuracies_validation = json.dumps(result.get('train_stats', {}).get('epoch_accuracies_validation', []))

            bias = result.get('train_stats', {}).get('final_loss', 0)
            test_loss = result.get('test_stats', {}).get('test_loss', 0)
            epochs_trained = result.get('train_stats', {}).get('epochs_trained', 0)
            model_architecture = result.get('model_architecture', None)
            error_message = result.get('error_message', None)

            # Prepare the data tuple, ensuring all fields are correctly set
            data_tuple = (
                test_id,
                exp_id,
                score,
                formatted_training_date,
                duration_seconds,
                hardware_config,

                epoch_losses_train,
                epoch_losses_validation,
                epoch_accuracies_train,
                epoch_accuracies_validation,
                bias,
                test_loss,
                epochs_trained,
                model_architecture,
                error_message
            )

            # Debugging: Print the data tuple
            print("Data Tuple:", data_tuple)

            # Update test record in the database
            query_test = """
                INSERT INTO test (test_id, exp_id, score, date, duration_seconds, hardware_config,
                                  epoch_losses_train, epoch_losses_validation, epoch_accuracies_train,
                                  epoch_accuracies_validation, bias, test_loss, epochs_trained,
                                  model_architecture, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                score=VALUES(score), date=VALUES(date), duration_seconds=VALUES(duration_seconds),
                hardware_config=VALUES(hardware_config), epoch_losses_train=VALUES(epoch_losses_train),
                epoch_losses_validation=VALUES(epoch_losses_validation), 
                epoch_accuracies_train=VALUES(epoch_accuracies_train),
                epoch_accuracies_validation=VALUES(epoch_accuracies_validation), bias=VALUES(bias),
                test_loss=VALUES(test_loss), epochs_trained=VALUES(epochs_trained), 
                model_architecture=VALUES(model_architecture),
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
