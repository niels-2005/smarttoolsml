import os


# AWS CLI needs to be configured.
def sync_folder_to_s3(folder, aws_bucket_url):
    """_summary_

    Args:
        folder (_type_): _description_
        aws_bucket_url (_type_): _description_
    Example usage:
        aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
        self.s3_sync.sync_folder_to_s3(
                folder=self.training_pipeline_config.artifact_dir,
                aws_bucket_url=aws_bucket_url,
            )
    """
    command = f"aws s3 sync {folder} {aws_bucket_url}"
    os.system(command)


def sync_folder_from_s3(folder, aws_bucket_url):
    command = f"aws s3 sync {aws_bucket_url} {folder}"
    os.system(command)
