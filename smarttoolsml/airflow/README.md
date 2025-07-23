To test specific tasks in the Airflow DAG, you can use the following command:

Go to the Docker Container where Airflow Scheduler is running and execute:

```bash
/bin/bash
```

```bash
airflow tasks test <dag_id> <task_id> <execution_date>
```

Example:

```bash
airflow tasks test user_processing_dag create_user_table
``` 

This command will run the `create_user_table` task in the `user_processing_dag` DAG for the specified execution date without affecting the actual DAG run. This is useful for debugging and testing individual tasks in isolation.