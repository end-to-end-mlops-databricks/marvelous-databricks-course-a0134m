from databricks.sdk import WorkspaceClient


def check_if_endpoint_exists(client, endpoint_name):
    try:
        client.serving_endpoints.get(endpoint_name)
        return True
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            return False
        # raise e Don't raise the exception, just return False so we can create the endpoint


def get_latest_entity_version(client, endpoint_name):
    serving_endpoint = client.serving_endpoints.get(endpoint_name)
    latest_version = 1
    for entity in serving_endpoint.config.served_entities:
        version_int = int(entity.entity_version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


def get_model_entity_index(client, endpoint_name, full_model_name) -> int:
    serving_endpoint = client.serving_endpoints.get(endpoint_name)
    model_index : int = -1 # endpoint only serving one model
    for idx, entity in enumerate(serving_endpoint.config.served_entities):
        if entity.entity_name == full_model_name:
            model_index = int(idx)
            break
    return model_index


def get_env_config_file(env:str) -> str:
    if env == "dev":
        return "project_config_dev.yml"
    elif env == "prod":
        return "project_config_prod.yml"
    else:
        raise ValueError(f"Invalid environment: {env}")

