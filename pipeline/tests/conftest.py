from __future__ import annotations

import pytest
import db as dbmod


@pytest.fixture
def conn():
    connection = dbmod.connect(type("Cfg", (), {"db_path": ":memory:"}))
    dbmod.init_schema(connection)
    yield connection
    connection.close()
