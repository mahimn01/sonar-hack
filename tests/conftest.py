import pytest

from sonar.config import SonarConfig, default_config


@pytest.fixture(scope="session")
def cfg() -> SonarConfig:
    return default_config()
