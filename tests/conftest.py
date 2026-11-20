"""Test configuration: override paywall for general API tests."""
import pytest
from fastapi.testclient import TestClient

from gradeforge.app import create_app
from gradeforge.config import Settings


@pytest.fixture()
def client(tmp_path):
    """Provides a TestClient with paywall disabled so tests can run unauthenticated."""
    return TestClient(create_app(tmp_path, Settings(paywall=False)))
