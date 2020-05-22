import os


def get_project_root():
    """
    Returns the root director of the project
    :return: The root director of the project
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
